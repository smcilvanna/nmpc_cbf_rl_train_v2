import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomSystemEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Action space: 21 continuous [0,1] values
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(21,),
            dtype=np.float32
        )
        
        # Observation space: 93 continuous values
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(93,),
            dtype=np.float32
        )
        
        # Initialize simulation state
        self.state = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        # Reset your simulation here
        self.state = np.zeros(93, dtype=np.float32)  # Replace with actual reset
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        # Convert first action element to discrete value
        discrete_action = int(np.round(action[0] * (NUM_DISCRETE_OPTIONS - 1)))
        discrete_action = np.clip(discrete_action, 0, NUM_DISCRETE_OPTIONS - 1)
        
        # Continuous actions (elements 1-20)
        continuous_actions = action[1:]
        
        # Call your existing simulation step here
        # REPLACE WITH YOUR SIMULATION CALL
        self.state = self._simulate_step(discrete_action, continuous_actions)
        
        # Calculate reward
        reward = self._calculate_reward()  # Implement your reward logic
        
        # Check termination (customize as needed)
        terminated = self.current_step >= 1000
        truncated = False
        
        self.current_step += 1
        return self.state, reward, terminated, truncated, {}

    def _simulate_step(self, discrete_action, continuous_actions):
        """Replace with your actual simulation call"""
        # Your existing code that advances the simulation 0.1s
        return np.random.randn(93).astype(np.float32)  # Example

    def _calculate_reward(self):
        """Replace with your actual reward calculation"""
        return 0.0  # Example
