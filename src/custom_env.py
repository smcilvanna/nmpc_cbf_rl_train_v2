import gymnasium as gym
from gymnasium import spaces
import numpy as np
from time import time
from generateCurriculumEnvironment import genCurEnv_2 as genenv2
from nmpc_cbf import NMPC_CBF_MULTI_N
from episodeTracker import EpisodeTracker

from testSim import getStepObservations, checkCollision, episodeTermination, calculate_reward

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
        self.currentPos = np.array([0.0,0.0,self.targetPos[2]])                     
        self.state = getStepObservations(                                           # get initial observation          
                        currentState=self.currentPos,
                        u=np.array([0.0,0.0]), 
                        mpcTime=0.00, 
                        env=self.map,
                        normalise=True)
        return self.state, {}   # return observation, no additional info

    def step(self, action):

        # Simulate Step
        newPos, u, mpcTime = self.simulateStep(self.currentPos, action[:-1])
        
        # Get Observations From Step
        self.state = getStepObservations(newPos, u, mpcTime, self.map, normalise=True)
        
        # Log observations and actions
        self.ep.add_observation(self.state)
        self.ep.add_action(action.flatten().tolist())

        # check if pass target is hit
        if len(self.gateCheck) > 0:
            for i, gate in enumerate(self.gateCheck):
                hitgate, _ = checkCollision(newPos, np.array(gate + [0.6]))
                if hitgate: # if hit
                    self.gateCheck.pop(i)
                    self.ep.epPassGates =+ 1
                    break
        
        # Check for termination conditions
        timeout = self.current_step >= self.maxSimSteps     # Check for timeout
        isdone = episodeTermination(self.ep)
        isdone[0] = isdone[0] or timeout                   # Add timeout check to isdone
        self.ep.done = isdone[0]
        if not timeout:
            reward = calculate_reward(self.ep,isdone)
        else:
            reward = -70

        # Check termination (customize as needed)
        terminated = isdone[0]
        truncated = False
        
        self.ep.add_reward(reward)                          # Log reward
        
        # advance for next step
        self.currentPos = newPos
        self.current_step += 1
        return self.state, reward, terminated, truncated, {}
    
    def simulateStep(self, startState , cbf):
        t = time()
        try:
            u = self.nmpc.solve(startState,cbf)
        except:
            print("Solver Fail")
            u = np.array([0,0]) # if no solver solution stop and let fail episode with low velocity
        currentPos = self.nmpc.stateHorizon[0,:]
        mpcTime = time() - t
        return currentPos, u, mpcTime


########## TEST #########

if __name__ == "__main__":

    env = CustomSystemEnv()
    obs, info = env.reset()

    action = np.ones((1,env.nmpc.nObs+1))*0.00
    for i in range(2000):
    # print(np.cos(env.currentPos[-1]))
        obs, rew, term, trun, info = env.step(action)
        print(obs[0:8], term)
        if term:
            break
    print(env.targetPos)
    print(env.ep.rewards)
    print(sum(env.ep.rewards))


