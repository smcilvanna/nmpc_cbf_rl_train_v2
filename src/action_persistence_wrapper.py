import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ActionPersistenceWrapper(gym.Wrapper):
    def __init__(self, env, persist_steps=10):
        super().__init__(env)
        self.persist_steps = persist_steps
        self.current_action = np.zeros(env.action_space.shape, dtype=np.float32)
        self.steps_since_update = 0
        
        # Augment observation space with temporal counter
        original_low = self.observation_space.low
        original_high = self.observation_space.high
        self.observation_space = spaces.Box(
            low=np.append(original_low, [0]),
            high=np.append(original_high, [1]),
            shape=(self.observation_space.shape[0] + 1,),
            dtype=np.float32
        )

    def step(self, action):
        # Update action only when required
        if self.steps_since_update % self.persist_steps == 0:
            self.current_action = action.copy()
        
        # Execute environment step
        obs, reward, terminated, truncated, info = self.env.step(self.current_action)
        
        # Augment observation with normalized step counter
        augmented_obs = np.append(obs, self.steps_since_update / self.persist_steps)
        
        # Update step counter
        self.steps_since_update = (self.steps_since_update + 1) % self.persist_steps
        
        return augmented_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.steps_since_update = 0
        self.current_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        obs, info = self.env.reset(**kwargs)
        return np.append(obs, 0.0), info