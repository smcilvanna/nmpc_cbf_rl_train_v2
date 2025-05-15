import gymnasium as gym
import numpy as np
from generateCurriculumEnvironment import genCurEnv_2
from nmpc_cbf import NMPC_CBF_MULTI_N
from time import time
from collections import deque

class ActionPersistenceWrapper(gym.Wrapper):
    def __init__(self, env, persist_steps=5):
        super().__init__(env)
        self.persist_steps = persist_steps
        self.current_action = 0  # Default first action
        self.steps_since_change = self.persist_steps # Initialize to trigger action on first step

    def step(self, action):
        if self.steps_since_change >= self.persist_steps:
            self.current_action = action
            self.steps_since_change = 0
        self.steps_since_change += 1
        return super().step(self.current_action)

    def reset(self, map=None, **kwargs):
        self.steps_since_change = self.persist_steps  # Reset to trigger action on first step
        self.current_action = 0  # Reset action to default
        return self.env.reset(map=map, **kwargs)


class MPCHorizonEnv(gym.Env):
    def __init__(self, curriculum_level=1):
        super().__init__()
        # self.last_horizon = None
        self.current_horizon = None
        self.last_target_dist = []
        # Curriculum parameters
        self.curriculum_level = curriculum_level
        self.horizon_options = list(range(10,151,10))  # 10-100 in steps of 10 #changed from 5
        self.past_lin_vels = deque(maxlen=5)
        self.av_lin_vel = 0

        # Action space: Discrete horizon selection
        self.action_space = gym.spaces.Discrete(len(self.horizon_options))
        
        # Observation space: [mpc_time, target_dist, target_sin, target_cos] + 20*(obs_dist, obs_sin, obs_cos) + (lin_vel ave_lin_vel)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(4 + 4*3 + 2,),
            dtype=np.float32
        )
        
        # MPC system
        self.nmpc = NMPC_CBF_MULTI_N(0.1, self.horizon_options, nObs=4)
        # self.reset()

    def add_velocity(self,v):
        self.past_lin_vels.append(v)
        self.av_lin_vel = sum(self.past_lin_vels)/len(self.past_lin_vels)
        return 

    def reset(self, map=None , seed=None, options=None):
        # Generate new environment
        if map == None:
            self.map = genCurEnv_2(curriculum_level=self.curriculum_level, 
                                       gen_fig=False, maxObs=self.nmpc.nObs)
        else:
            print('Loading Saved Map')
            self.map = map
        # Initialize MPC
        self.nmpc.setObstacles(self.map['obstacles'])
        self.nmpc.setTarget(self.map['target_pos'])
        self.current_pos = np.array([0.0, 0.0, self.map['target_pos'][2]])
        self.nmpc.reset_nmpc(self.current_pos)
        self.cbf_per_obs = self.get_cbf_values(self.map['obstacles'])
        init_obs, _ = self._get_obs()


        # Set pass target point for calcs
        self.pass_tgt = np.array(self.map["pass_targets"]).flatten()

        # Set obs mid point for calcs
        oxy1 = self.map["obstacles"][0,0:2]
        oxy2 = self.map["obstacles"][1,0:2]
        self.mid = (oxy1 + oxy2)/2

        # Calculate angle between deadlock and pass points
        self.radang = abs(np.arctan2(self.pass_tgt[1], self.pass_tgt[0]) - np.arctan2(self.mid[1], self.mid[0]))
        self.passpoint = False
        return init_obs, {}

    def get_cbf_values(self,obstacles):
        cbfs = []
        for orad in obstacles[:,2]:
            cbf = 1.0742 * np.exp(-2.0 * orad) + 0.0225 
            cbfs.extend([cbf])
        return np.array(cbfs).reshape((1,-1))

    def _get_obs(self, mpc_time=0.0):
        """Simplified observation vector"""
        # Initialize as list
        obs_list = []
        
        # 1. MPC performance
        obs_list.append(mpc_time)
        
        # 2. Target information
        target_vec = self.map['target_pos'][:2] - self.current_pos[:2]
        target_dist = np.linalg.norm(target_vec)
        target_angle = np.arctan2(target_vec[1], target_vec[0])
        obs_list.extend([target_dist, np.sin(target_angle), np.cos(target_angle)])
        
        # 3. Obstacle information (20 obstacles)
        min_obs_dist = 1000
        for obstacle in self.map['obstacles']:  # Renamed variable to avoid conflict
            vec = obstacle[:2] - self.current_pos[:2]
            dist = np.linalg.norm(vec) - 0.55 - obstacle[2]
            min_obs_dist = dist if dist < min_obs_dist else min_obs_dist
            angle = np.arctan2(vec[1], vec[0])
            obs_list.extend([dist, np.sin(angle), np.cos(angle)])
        # print(f"Closest Obstacle : {min_obs_dist} m")

        # 4. Velocities, current average
        if len(self.past_lin_vels) > 0:
            obs_list.extend([self.past_lin_vels[-1]])
            obs_list.extend([self.av_lin_vel])
        else:
            obs_list.extend([0.0,0.0])

        # Convert to numpy array at the end
        self.min_obs_dist = min_obs_dist
        return np.array(obs_list, dtype=np.float32), min_obs_dist

    def step(self, action):
        # If horizon action changes current horizon adjust it in nmpc
        if self.current_horizon != self.horizon_options[action]:
            self.current_horizon = self.horizon_options[action]
            self.nmpc.adjustHorizon(self.current_horizon, self.current_pos)
            # print("Horizon Changed")
        # horizon_changed = True if (self.current_horizon == self.last_horizon or self.last_horizon == None) else False
        # Solve MPC  <<<<<<<<<<<<<< ADD CBF CUSTOM PREDICT HERE
        t = time()
        try:
            u = self.nmpc.solve(self.current_pos, self.cbf_per_obs)
            mpc_time = time() - t
        except:
            print("[WARN] Solver Fail Controller Output Zeroed")
            u = np.zeros(2)
            mpc_time = 0.5
        # Update state and velocity
        self.current_pos = self.nmpc.stateHorizon[0,:]
        self.add_velocity(u[0])
        
        observations, min_obs_dist = self._get_obs(mpc_time)

        # Calculate reward and done
        reward, done = self._calculate_reward(self.current_pos, mpc_time, min_obs_dist)
        # self.last_mpc_time = mpc_time
        # self.last_horizon = self.current_horizon

        info = {
            "u": u,
            "mpc_time": mpc_time,
            "horizon": self.current_horizon
            # "collision": any(self.checkCollision(self.current_pos, obs)[0] for obs in self.map['obstacles']),
            # "target_distance": np.linalg.norm(self.current_pos[:2] - self.map['target_pos'][:2])
            }

        return observations, reward, done, False, info

    def _calculate_reward(self, position, mpc_time, min_obs_dist):
        target_dist = np.linalg.norm(position[:2] - self.map['target_pos'][:2])
        collision = any(self.checkCollision(position, obs)[0] for obs in self.map['obstacles'])
        velocity = self.past_lin_vels[-1]  # Assuming position[3] contains velocity (add to observations)
        
        # Velocity rewards (maintain ~1 m/s)
        velocity_reward = np.exp(-2*(velocity - 1.0)**2)  # Gaussian peak at 1 m/s
        velocity_reward *= 0.15
        # # MPC time rewards (piecewise function)
        # if mpc_time <= 0.05:
        #     time_reward = 1.0  # Max reward for fastest solves
        # elif mpc_time <= 0.1:
        #     # Linear increase: 0.1s → 1.0, 0.05s → 5.0
        #     time_reward = (0.1 - mpc_time)/(0.1 - 0.05)
        # elif mpc_time <= 0.25:
        #     # Linear decay: 0.1s → 1.0 → 0.25s → 0.0
        #     time_reward = 1.0 - (mpc_time - 0.1)/0.15  # Adjusted slope
        # elif mpc_time <= 1.0:
        #     # Linear penalty: 0.25s → 0.0 → 1.0s → -3.0
        #     time_reward = -3.0 * (mpc_time - 0.25)/0.75
        #     time_reward *= 2
        # else:
        #     # Constant penalty beyond 1s
        #     time_reward = -2.0
        if mpc_time <= 0.05:  # <50ms
            time_reward = 1.0
        elif mpc_time <= 0.15:  # 50-150ms
            # Linear decrease from 1.0 to 0.0 over 0.05-0.15s
            time_reward = 1.0 - ((mpc_time - 0.05) / 0.10)
        elif mpc_time <= 0.8:  # 150-800ms
            # Linear decrease from 0.0 to -2.0 over 0.15-0.8s
            time_reward = -2.0 * ((mpc_time - 0.15) / 0.65)
        else:  # >800ms
            time_reward = -2.0
        if time_reward < 0:
            time_reward *= 0.5

        # if time_reward > 0:
        #     time_reward = time_reward/4

        # Horizon efficiency bonus (encourage minimal sufficient horizons)
        # horizon_efficiency = 0.2 * (1 / (horizon/10)) if horizon < 50 else 0.0
        
        # Collision penalty (keep severe)
        collision_penalty = 500.0 if collision else 0.0
        
        # Progress reward (keep small since MPC handles progress)
        progress_reward = 0.05 * (self.last_target_dist - target_dist) if self.last_target_dist else 0.0
        
        # Deadlock penalty - if average velocity falls too low
        deadlock_penalty = 500.0 if len(self.past_lin_vels) >= 5 and self.av_lin_vel < 0.2 else 0
        # increase penalty if short horizon
        # if deadlock_penalty > 0 and self.current_horizon < 41:
        #     deadlock_penalty *= 1 + (40 - self.current_horizon)/30
        
        # Smoothness penalty (discourage frequent horizon changes)
        # change_penalty = 0.8 if horizon_changed else 0.0
        
        # Horizon bonus close to obstacles
        obs_threshold = 3.0
        # horizon_bonus = 0.5*self.current_horizon*(obs_threshold - min_obs_dist) if min_obs_dist < obs_threshold else 0.0
        # Also reduce any time penalties near obstacle for long horizon
        if min_obs_dist < obs_threshold: #and time_reward < 0:
            reward_mul = max(0.2, 0.32*min_obs_dist+0.04)
            # print(reward_mul)
            time_reward *= reward_mul

        # Check and report terminal conditions
        at_target = target_dist < 0.5
        deadlock = True if deadlock_penalty > 0 else False

        # Before obstacle assign reward / penalty for aiming at pass/deadlock points
        aim_reward = 0
        if not self.passpoint and self.min_obs_dist > 0.05:

            # Check if passpoint has been reached
            vec = self.pass_tgt[:2] - self.current_pos[:2]
            self.passpoint = (np.linalg.norm(vec) - 0.75) < 0.0

            # Check if aiming at deadlock
            vec = self.mid - self.current_pos[:2]
            dang = np.arctan2(vec[1], vec[0])
            dead_yaw = abs(dang - self.current_pos[2]) < self.radang

            # Check if aiming for passpoint
            vec = self.pass_tgt - self.current_pos[:2]
            pang = np.arctan2(vec[1], vec[0])
            pass_yaw = abs(pang - self.current_pos[2]) < self.radang


            if pass_yaw:
                # print("Aiming to clear")
                aim_reward = 1.5
            if dead_yaw:
                # print("Aiming at deadlock")
                aim_reward = -1.5
                

        if deadlock:    
            print(f"[FAIL] Vehicle Deadlock | Average Velocity: {self.av_lin_vel}m/s across {len(self.past_lin_vels)} samples")
        if collision:
            print(f"[FAIL] Collision")
        if at_target:
            print(f"[SUCCESS] At target!")
            progress_reward += 200  # increased reward from 10 to reward successful navigation

        done =  at_target or collision or deadlock
        self.last_target_dist = target_dist
        
        total_reward = (
            velocity_reward
            + time_reward
            # horizon_efficiency +
            + progress_reward
            - collision_penalty
            - deadlock_penalty
            # - change_penalty
            + aim_reward
        )
        # print("total : ",total_reward)
        # print("vel   : ",velocity_reward)
        # print("time  : ", time_reward, "for ", round(mpc_time,4))
        # print("p     : ", progress_reward)
        return total_reward, done
    
    def checkCollision(self,vehiclePos, obs):
        cen_sep = np.linalg.norm(vehiclePos[0:2] - obs[0:2])
        safe_sep = cen_sep - 0.55 - obs[2]
        collision = safe_sep <= 0.0 
        return collision, safe_sep

if __name__ == "__main__":
    print("Custom Environment ClassDef")


