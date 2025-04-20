
import numpy as np
from generateCurriculumEnvironment import generate_curriculum_environment as genenv
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from nmpc_cbf import NMPC_CBF_MULTI_N

np.set_printoptions(precision=3, suppress=True)

def plotSimdata(simdata,target):
    fig = plt.figure(figsize=(10, 6))
    
    ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)    # ax1: left, spanning 2 rows and 2 columns  
    ax2 = plt.subplot2grid((2, 4), (0, 2), rowspan=1, colspan=2)    # ax2: top right, 1 row, 2 columns
    ax3 = plt.subplot2grid((2, 4), (1, 2), rowspan=1, colspan=2)    # ax3: bottom right, 1 row, 2 columns

    
    # Position Plot
    t = simdata[:,0]
    mpct = simdata[:,1]*1000
    x = simdata[:,2]
    y = simdata[:,3]
    th = simdata[:,4]
    v = simdata[:,5]
    w = simdata[:,6]


    ax1.scatter(x, y,s=1)      # Plots y versus x as a line
    ax1.add_patch(Circle(simdata[-1,2:5], 0.55, color='black', alpha=0.9, label="vehicle"))
    ax1.add_patch(Circle(target[0:2], 0.2, color='green', alpha=0.9))
    for i in range(obstacles.shape[0]):
        ax1.add_patch(Circle( obstacles[i,0:2], obstacles[i,2], color='red')) 

    ax2.plot(t,mpct)
    ax3.plot(t,v, label="Linear (m/s)")
    ax3.plot(t,w, label="Angular (rad/s)")

    # Set axis limits for ax1
    lim = np.max(target[0:2])
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)

    # Set axis labels
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Solve Time (ms)')
    ax3.set_xlabel('Simulation Step')
    ax3.set_ylabel('Velocity')
    ax3.legend()

    plt.tight_layout()
    plt.show()

def simulateStep(stepTime, startState, obstacles ):
    simSteps = int(stepTime / 0.1)          # hardcoded timestep dt = 0.1
    print(simSteps)
    simdata = np.zeros(( simSteps + 1, 7))  # [stepIndex solveTime x y yaw v w]
    simdata[0,2:5] = startState
    currentPos = startState
    for i in range(1,simSteps+1):
        t = time.time()
        u = nmpc.solve(targetPos,currentPos,obstacles,cbf)
        currentPos = nmpc.stateHorizon[1,:]
        simdata[i,0] = i
        simdata[i,1] = time.time() - t
        simdata[i,2:5] = currentPos
        simdata[i,5:] = u
        print(f"{i} : {u}")
    return simdata         
    

if __name__ == "__main__":

    env = genenv(5, gen_fig=False)
    # plt.show()
    # print(env["obstacles"])
    # print(env)
    # exit()

    nmpc = NMPC_CBF_MULTI_N(0.1, [10, 20, 30, 40, 50], 6)
    print("NMPC_CBF_MULTI_N class initialized successfully.")
    nmpc.solversIdx = 0
    nmpc.currentN = nmpc.nVals[nmpc.solversIdx]
    obstacles = env['obstacles']
    targetPos = env['target_pos']
    print(targetPos)
    print(np.rad2deg(targetPos[2]))

    startPos = np.array([0,0,targetPos[2]])
    cbf = np.array([ 1,1,1,1,1,1])*0.8

    simdata = simulateStep(5,startPos,obstacles)

    # print(simdata)
    plotSimdata(simdata,targetPos)
    # print(simdata)

    # startPos = simdata[-1,2:5]
    # simdata = simulateStep(10,startPos,obstacles)
    # plotSimdata(simdata,targetPos)

