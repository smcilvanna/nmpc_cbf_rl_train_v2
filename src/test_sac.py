import numpy as np
from generateCurriculumEnvironment import genCurEnv_2 as genenv2
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import pickle

from nmpc_cbf import NMPC_CBF_MULTI_N
from episodeTracker import EpisodeTracker

from testSim import simulateStep, getStepObservations, checkCollision, episodeTermination, calculate_reward



if __name__ == "__main__":
    print("[START]")
    random_env = True
    if random_env:
        # env = genenv(2, gen_fig=True)
        # plt.show()
        # input("[ENTER] to begin")
        env = genenv2(curriculum_level=1,gen_fig=True,maxObs=20)
        plt.show()
    else:
        file_path = './env1-1.pkl'
        with open(file_path, 'rb') as f:
            env = pickle.load(f)
    # exit()
    # Nvalues = [10,20,30,40,50,60,70,80,90,100] #np.arange(10,110,10)#[10, 20, 30, 40, 50]
    Nvalues = [10 , 20]
    nmpc = NMPC_CBF_MULTI_N(0.1, Nvalues, nObs=20)
    print("NMPC_CBF_MULTI_N class initialized successfully.")
    
    # Set initial mpc parameters
    # nmpc.solversIdx = nmpc.normalActionsN(np.random.uniform(0,1)) # random start solver index
    nmpc.solversIdx = 1
    print(nmpc.solversIdx)
    nmpc.currentN = nmpc.nVals[nmpc.solversIdx]         # random start solver N
    obstacles = env['obstacles']                        # obstacle config from environment
    obstacles = obstacles[0:nmpc.nObs,:]
    targetPos = env['target_pos']                       # target position from environment
    nmpc.setObstacles(obstacles)
    nmpc.setTarget(targetPos)
    
    currentPos = np.array([0,0,targetPos[2]])
    targetArea = np.append(targetPos,0.05)
    
    # cbf = np.tile(5e-2,nmpc.nObs)
    # cbf = np.random.uniform(0,1, size=(1,nmpc.nObs))
    cbf = np.ones((1,nmpc.nObs))*0.01
    print(nmpc.normalActionsCBF(cbf))
    ep = EpisodeTracker(allRecord=False)
    cnt=0
    gateCheck = env["pass_targets"].copy()
    totalReward = 0
    maxSimTime = env["startDist"]*2
    maxSimSteps = int(maxSimTime / nmpc.dt)
    sd = env["startDist"]
    input(f"Start Target Distance : {sd:.2f} m\nMaximum Sim Time : {maxSimTime:.1f} seconds\nMax Episode Steps : {maxSimSteps} ")
    while not ep.done:
         
        newPos, u, mpcTime = simulateStep(currentPos, cbf)        
        observe = getStepObservations(newPos,u,mpcTime,env)     # get observations for next step
        ep.add_observation(observe)                             # update observations for episode
        actions = cbf.flatten().tolist()
        actions.append(nmpc.currentN)
        ep.add_action(actions)

        # check if pass target is hit
        if len(gateCheck) > 0:
            for i, gate in enumerate(gateCheck):
                hitgate, _ = checkCollision(newPos, np.array(gate + [0.6]))
                if hitgate: # if hit
                    gateCheck.pop(i)
                    ep.epPassGates =+ 1
                    break

        isdone = episodeTermination(ep)
        ep.done = isdone[0]
        ep.add_reward(calculate_reward(ep,isdone))
        print(ep.rewards[-1])
        
        # advance for next step
        currentPos = newPos
        
        cnt = cnt+1
        if cnt % 10 == 0:
            # nmpc.adjustHorizon(np.random.uniform(0,1))
            # cbf = np.random.uniform(0,1, size=(1,nmpc.nObs))
            continue
        if cnt > maxSimSteps:
            print("EPISODE TIMEOUT")
            ep.done = True
    