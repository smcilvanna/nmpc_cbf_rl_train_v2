import gymnasium as gym
from gymnasium import spaces
import numpy as np

from generateCurriculumEnvironment import genCurEnv_2 as genenv2
from nmpc_cbf import NMPC_CBF_MULTI_N
from episodeTracker import EpisodeTracker

from testSim import getStepObservations

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
        self.current_step = 0                                               # step counter
        self.ep = EpisodeTracker(allRecord=False)
        self.nmpc = NMPC_CBF_MULTI_N(0.1, list(range(10, 101, 5)), nObs=20) # create solvers
        # self.reset()

    def reset(self, seed=None, options=None):
        
        self.ep.reset()                                                             # reset episode tracker
        self.map = genenv2(curriculum_level=1,gen_fig=False, maxObs=self.nmpc.nObs) # generate random map for episode
        self.nmpc.setObstacles(self.map['obstacles'])                               # set obstacles for solver
        self.targetPos = self.map['target_pos']                                     
        self.nmpc.setTarget(self.targetPos)                                         # set target for solver
        self.maxSimSteps = int(self.map["startDist"]*2 / self.nmpc.dt)              # calculate maximum sim time = 2*dist to target @ 1m/s
        self.gateCheck = self.map["pass_targets"].copy()                            # copy target gates for reward checking
        self.current_step = 0                                                       # reset step counter
        self.state = getStepObservations(np.array([0,0,self.targetPos[2]]),         # get initial observation
                                         np.array[0,0], 
                                         0.00, 
                                         self.map)
        return self.state, {}   # return observation, no additional info

    def step(self, action):
        # # Convert first action element to discrete value
        # discrete_action = int(np.round(action[0] * (NUM_DISCRETE_OPTIONS - 1)))
        # discrete_action = np.clip(discrete_action, 0, NUM_DISCRETE_OPTIONS - 1)
        
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


########## TEST #########

if __name__ == "__main__":

    env = CustomSystemEnv()
    obs, info = env.reset()
    
    print("Initial observation shape:", obs.shape)
    print("State array:", env.state)