import gymnasium as gym
import numpy as np
from generateCurriculumEnvironment import genCurEnv_2
from nmpc_cbf import NMPC_CBF_MULTI_N
from time import time
from collections import deque

# class ActionPersistenceWrapper(gym.Wrapper):
#     def __init__(self, env, persist_steps=5):
#         super().__init__(env)
#         self.persist_steps = persist_steps
#         self.current_action = 0  # Default first action
#         self.steps_since_change = self.persist_steps # Initialize to trigger action on first step

#     def step(self, action):
#         if self.steps_since_change >= self.persist_steps:
#             self.current_action = action
#             self.steps_since_change = 0
#         self.steps_since_change += 1
#         return super().step(self.current_action)

#     def reset(self, **kwargs):
#         self.steps_since_change = self.persist_steps  # Reset to trigger action on first step
#         self.current_action = 0  # Reset action to default
#         return super().reset(**kwargs)


class MPCHorizonEnv(gym.Env):
    def __init__(self, curriculum_level=1):
        super().__init__()
        # self.last_horizon = None
        self.current_horizon = None
        self.last_target_dist = []
        self.last_action = np.zeros(1)
        self.obstacle_attention = 1

        # Curriculum parameters
        self.curriculum_level = curriculum_level
        self.horizon_options = [30]  # 10-100 in steps of 5
        self.past_lin_vels = deque(maxlen=10)
        self.av_lin_vel = 0

        # Action space: CBF parameter values for 3 obstacles
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obstacle_attention,),
            dtype=np.float32
        )
        # 
        #
        #                       0           1           2           3                  4         5        6        7          8        9          10       11
        # Observation space: [mpc_time, target_dist, target_sin, target_cos] + obs*(obs_dist, obs_sin, obs_cos  obs_rad) + (lin_vel ave_lin_vel, sin(yaw) cos(yaw))
        # Limits                1s         50m                                      20m                         10m         
        # Normalised to        [0 1]      [0 1]       [-1 1]       [-1 1]          [0 1]     [-1 1]   [-1 1]   [0 1]       [0 1]    [0 1 ]      [-1 1]    [-1 1]
        
        obs_size = 1 + 3 + (4 * self.obstacle_attention) + 2 + 2
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # MPC system
        self.nmpc = NMPC_CBF_MULTI_N(0.1, self.horizon_options, nObs=self.obstacle_attention)
        self.veh_rad = self.nmpc.vehRad
        # self.reset()

    def set_curriculum_level(self, level):
        """Update curriculum level dynamically."""
        self.curriculum_level = level

    def add_velocity(self,v):
        self.past_lin_vels.append(v)
        self.av_lin_vel = sum(self.past_lin_vels)/len(self.past_lin_vels)
        return 

    def reset(self, seed=None, options=None):
        # Generate new environment
        self.map = genCurEnv_2(curriculum_level=self.curriculum_level, 
                              gen_fig=False, maxObs=20)
        # Initialize MPC
        self.nmpc.setTarget(self.map['target_pos'])
        self.current_pos = np.array([0.0, 0.0, self.map['target_pos'][2]])
        self.nmpc.reset_nmpc(self.current_pos)
        self.obstacle_check = np.hstack((self.map["obstacles"], np.zeros((20, 1))))
        self._update_closest_obstacles()
        return self._get_obs(), {}

    def _update_closest_obstacles(self):
        # Check all obstacles in environment
        for i, _ in enumerate(self.obstacle_check):
            vec = self.obstacle_check[i,:2] - self.current_pos[:2]
            self.obstacle_check[i,3] = np.linalg.norm(vec) - self.veh_rad - self.obstacle_check[i,2]
        obs_sorted = self.obstacle_check[self.obstacle_check[:,3].argsort()]
        self.closest_obstacles = obs_sorted[0:self.nmpc.nObs , 0:3]
        self.nmpc.setObstacles(self.closest_obstacles)


    def _get_obs(self, mpc_time=0.0):
        """Simplified observation vector"""
        # Initialize as list
        obs_list = []
        
        # 1. MPC performance [0]
        obs_list.append(np.clip(mpc_time,0.0,1.0))  # Normalised [0-1] max 1 second
        
        # 2. Target information [1:4]
        target_vec = self.map['target_pos'][:2] - self.current_pos[:2]
        target_dist = np.linalg.norm(target_vec)
        target_dist = np.clip(target_dist/50,0.0,1.0)   # Normalised [0-1] max 50m
        target_angle = np.arctan2(target_vec[1], target_vec[0])
        obs_list.extend([target_dist, np.sin(target_angle), np.cos(target_angle)])
        
        # 3. Obstacle information (3 obstacles) [4:13]
        for obstacle in self.closest_obstacles:
            vec = obstacle[:2] - self.current_pos[:2]
            dist = np.linalg.norm(vec) - self.veh_rad - obstacle[2]
            dist = np.clip(dist/50, 0.0, 1.0)   # Normalised [0-1] max 50 m
            angle = np.arctan2(vec[1], vec[0])
            obs_list.extend([dist, np.sin(angle), np.cos(angle), obstacle[2]])

        # 4. Velocities, current average [13:15]
        if len(self.past_lin_vels) > 0:
            obs_list.extend([self.past_lin_vels[-1]])
            obs_list.extend([self.av_lin_vel])
        else:
            obs_list.extend([0.0,0.0])

        # 5. Current yaw angle
        obs_list.extend([np.sin(self.current_pos[2]), np.cos(self.current_pos[2])])

        # Convert to numpy array at the end
        return np.array(obs_list, dtype=np.float32)

    def step(self, action):
        self._update_closest_obstacles()

        # Solve MPC
        t = time()
        try:
            u = self.nmpc.solve(self.current_pos, action)
        except:
            print("[WARN] Solver Fail Controller Output Zeroed")
            u = np.zeros(2)
        mpc_time = time() - t

        # Update state and velocity
        self.current_pos = self.nmpc.stateHorizon[0,:]
        self.add_velocity(u[0])
        
        # Calculate reward and done
        reward, done = self._calculate_reward(self.current_pos, mpc_time, action, u)

        info = {
            "u": u
        }
        #     "mpc_time": mpc_time,
        #     "horizon": self.current_horizon
        #     # "collision": any(self.checkCollision(self.current_pos, obs)[0] for obs in self.map['obstacles']),
        #     # "target_distance": np.linalg.norm(self.current_pos[:2] - self.map['target_pos'][:2])
        #     }

        return self._get_obs(mpc_time), reward, done, False, info

    def _calculate_reward(self, position, mpc_time, action, u):
        
        # Velocity rewards (maintain ~1 m/s)
        velocity = self.past_lin_vels[-1]
        velocity_reward = np.exp(-2*(velocity - 1.0)**2)  # Gaussian peak at 1 m/s
        velocity_reward = velocity_reward*2 -1

        if u[0] == 0.0:
            solverpen = 10.0
        else:
            solverpen = 0.0
        
        # MPC time rewards (piecewise function)
        if mpc_time <= 0.025:
            time_reward = 5.0  # Max reward for fastest solves
        elif mpc_time <= 0.50:
            time_reward = -3.0 + (8.0) * (0.500 - mpc_time)/(0.5-0.025)
        else:
            time_reward = -3.0

        if time_reward > 0:
            time_reward /= 4
        elif time_reward < 0:
            time_reward /= 2

        # Check obstacle seperations
        min_sep = 1e4
        for obstacle in self.closest_obstacles:
            vec = obstacle[:2] - self.current_pos[:2]
            sep = np.linalg.norm(vec) - self.veh_rad - obstacle[2]
            if sep < min_sep:
                min_sep = sep

        if min_sep <= 0.1:
            close_penalty = max(-3.0 , -25.0* max(0, (0.105-min_sep)))
        else:
            close_penalty = 0.0

        # Collision penalty (keep severe)
        collision = min_sep <= 0.0
        collision_penalty = 2000.0 if collision else 0.0
        
        # Progress reward
        target_dist = np.linalg.norm(position[:2] - self.map['target_pos'][:2])
        progress_reward = 1.0 * (self.last_target_dist - target_dist) if self.last_target_dist else 0.0
        
        # Deadlock penalty - if average velocity falls too low
        deadlock_penalty = 1000.0 if len(self.past_lin_vels) >= 10 and self.av_lin_vel < 0.05 else 0
        
        # Parameter change penalty
        param_change_penalty = 10 * np.linalg.norm(action - self.last_action, ord=2) if self.last_action != 0.0 else 0.0
        self.last_action = action.copy()


        # Check and report terminal conditions
        at_target = target_dist < 0.5
        deadlock = True if deadlock_penalty > 0 else False

        if deadlock:    
            print(f"[FAIL] Vehicle Deadlock | Average Velocity: {self.av_lin_vel}m/s across {len(self.past_lin_vels)} samples")
        if collision:
            print(f"[FAIL] Collision")
        if at_target:
            print(f"[SUCCESS] At target!")
            progress_reward += 15

        done =  at_target or collision or deadlock
        self.last_target_dist = target_dist
        
        total_reward = (
            velocity_reward
            + time_reward
            + close_penalty
            + progress_reward
            - collision_penalty
            - deadlock_penalty
            - param_change_penalty
            - solverpen
        )
        return total_reward, done
    
    def checkCollision(self,vehiclePos, obs):
        cen_sep = np.linalg.norm(vehiclePos[0:2] - obs[0:2])
        safe_sep = cen_sep - 0.55 - obs[2]
        collision = safe_sep <= 0.0 
        return collision, safe_sep

if __name__ == "__main__":
    print("Custom Environment ClassDef")