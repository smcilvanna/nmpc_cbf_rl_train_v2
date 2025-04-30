import gymnasium as gym
import numpy as np
from generateCurriculumEnvironment import genCurEnv_2
from nmpc_cbf import NMPC_CBF_MULTI_N
from time import time

class ActionPersistenceWrapper(gym.Wrapper):
    def __init__(self, env, persist_steps=5):
        super().__init__(env)
        self.persist_steps = persist_steps
        self.current_action = 0  # Default first action
        self.steps_since_change = 0

    def step(self, action):
        if self.steps_since_change >= self.persist_steps:
            self.current_action = action
            self.steps_since_change = 0
            
        self.steps_since_change += 1
        return super().step(self.current_action)

    def reset(self, **kwargs):
        self.steps_since_change = 0
        return super().reset(**kwargs)


class MPCHorizonEnv(gym.Env):
    def __init__(self, curriculum_level=1, action_interval=5):
        super().__init__()
        

        # Action interval parameter
        self.action_interval = action_interval  
        self.steps_since_action = 0
        self.last_horizon = None
        self.last_target_dist = []
        # Curriculum parameters
        self.curriculum_level = curriculum_level
        self.horizon_options = list(range(10, 101, 5))  # 10-100 in steps of 5

        # Action space: Discrete horizon selection
        self.action_space = gym.spaces.Discrete(len(self.horizon_options))
        
        # Observation space: [mpc_time, target_dist, target_sin, target_cos] + 20*(obs_dist, obs_sin, obs_cos)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(4 + 20*3,),
            dtype=np.float32
        )
        
        # MPC system
        self.nmpc = NMPC_CBF_MULTI_N(0.1, self.horizon_options, nObs=20)
        self.reset()

    def reset(self, seed=None, options=None):
        self.steps_since_action = 0
        self.last_horizon = None

        # Generate new environment
        self.map = genCurEnv_2(curriculum_level=self.curriculum_level, 
                              gen_fig=False, maxObs=self.nmpc.nObs)
        
        # Initialize MPC
        self.nmpc.setObstacles(self.map['obstacles'])
        self.nmpc.setTarget(self.map['target_pos'])
        self.current_pos = np.array([0.0, 0.0, self.map['target_pos'][2]])
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Simplified observation vector"""
        # Initialize as list
        obs_list = []
        
        # 1. MPC performance
        obs_list.append(self.last_mpc_time if hasattr(self, 'last_mpc_time') else 0.0)
        
        # 2. Target information
        target_vec = self.map['target_pos'][:2] - self.current_pos[:2]
        target_dist = np.linalg.norm(target_vec)
        target_angle = np.arctan2(target_vec[1], target_vec[0])
        obs_list.extend([target_dist, np.sin(target_angle), np.cos(target_angle)])
        
        # 3. Obstacle information (20 obstacles)
        for obstacle in self.map['obstacles']:  # Renamed variable to avoid conflict
            vec = obstacle[:2] - self.current_pos[:2]
            dist = np.linalg.norm(vec) - 0.55 - obstacle[2]
            angle = np.arctan2(vec[1], vec[0])
            obs_list.extend([dist, np.sin(angle), np.cos(angle)])
        
        # Convert to numpy array at the end
        return np.array(obs_list, dtype=np.float32)

    def step(self, action):
        # Only change horizon at specified intervals
        if self.steps_since_action % self.action_interval == 0:
            new_horizon = self.horizon_options[action]
            self.nmpc.adjustHorizon(new_horizon)
            self.last_horizon = new_horizon
            horizon_changed = True
        else:
            new_horizon = self.last_horizon
            horizon_changed = False
            
        self.steps_since_action += 1
        
        # Solve MPC  <<<<<<<<<<<<<< ADD CBF CUSTOM PREDICT HERE
        t = time()
        u = self.nmpc.solve(self.current_pos, np.ones(20)*0.5)
        self.current_pos = self.nmpc.stateHorizon[0,:]
        mpc_time = time() - t

        # Calculate reward with smoothness component
        reward, done = self._calculate_reward(self.current_pos, mpc_time, new_horizon, horizon_changed)
        
        # Update state
        self.last_mpc_time = mpc_time
        
        return self._get_obs(), reward, done, False, {"u":u}

    def _calculate_reward(self, position, mpc_time, horizon, horizon_changed):
        target_dist = np.linalg.norm(position[:2] - self.map['target_pos'][:2])
        collision = any(self.checkCollision(position, obs)[0] for obs in self.map['obstacles'])
        velocity = position[3]  # Assuming position[3] contains velocity (add to observations)
        
        # Velocity rewards (maintain ~1 m/s)
        velocity_reward = np.exp(-2*(velocity - 1.0)**2)  # Gaussian peak at 1 m/s
        deadlock_penalty = np.where(velocity < 0.1, 5.0, 0.0)  # Penalize near-zero velocity
        
        # MPC time rewards (piecewise function)
        if mpc_time <= 0.1:
            time_reward = 1.0  # Full reward for fast solves
        elif mpc_time <= 0.25:
            time_reward = 0.5 - (mpc_time - 0.1)/0.3  # Linear decay 0.5->0
        else:
            time_reward = -1.0  # Penalize excessive solve times
            
        # Horizon efficiency bonus (encourage minimal sufficient horizons)
        horizon_efficiency = 0.2 * (1 / (horizon/10)) if horizon < 50 else 0.0
        
        # Collision penalty (keep severe)
        collision_penalty = 100.0 if collision else 0.0
        
        # Progress reward (keep small since MPC handles progress)
        progress_reward = 0.5 * (self.last_target_dist - target_dist) if self.last_target_dist else 0.0
        
        # Smoothness penalty (discourage frequent horizon changes)
        change_penalty = 0.3 if horizon_changed else 0.0
        
        total_reward = (
            velocity_reward +
            time_reward +
            horizon_efficiency +
            progress_reward -
            collision_penalty -
            deadlock_penalty -
            change_penalty
        )
        
        done = target_dist < 0.5 or collision or velocity < 0.05
        self.last_target_dist = target_dist
        
        return total_reward, done
    
    def checkCollision(self,vehiclePos, obs):
        cen_sep = np.linalg.norm(vehiclePos[0:2] - obs[0:2])
        safe_sep = cen_sep - 0.55 - obs[2]
        collision = safe_sep <= 0.0 
        return collision, safe_sep



