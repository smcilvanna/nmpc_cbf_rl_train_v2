
import numpy as np
from generateCurriculumEnvironment import generate_curriculum_environment as genenv
import time
from matplotlib import pyplot as plt
from nmpc_cbf import NMPC_CBF_MULTI_N

np.set_printoptions(precision=3, suppress=True)

def plotSimdata(simdata):
    fig, ax = plt.subplots()
    x = simdata[:,2]
    y = simdata[:,3]
    ax.plot(x, y)      # Plots y versus x as a line

    for i in range(obstacles.shape[0]):
        circle = plt.Circle((obstacles[i,0:2]), radius=obstacles[i,2], color='red', fill=False)  # fill=False for outline only
        ax.add_artist(circle)
        # circle = plt.Circle((obstacles[1,0:2]), radius=1, color='red', fill=False)  # fill=False for outline only
        # ax.add_artist(circle)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y vs x')
    plt.show()


if __name__ == "__main__":

    env = genenv(2, gen_fig=True)
    plt.show()
    print(env["obstacles"])
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

    currentPos = np.array([0,0,targetPos[2]])
    cbf = np.array([ 0.1  , 0.1 ,0.1 , 0.9, 0.9, 0.9])

    simRealTime = 15
    simSteps = int(simRealTime / 0.1)
    simdata = np.zeros(( simSteps + 1, 5))
    # input(f"Simulate {simRealTime} seconds, {simSteps} steps")
    simdata[0,2:] = currentPos
    for i in range(simSteps):
        t = time.time()
        u = nmpc.solve(targetPos,currentPos,obstacles,cbf)
        currentPos = nmpc.stateHorizon[1,:]
        simdata[i,0] = i
        simdata[i,1] = time.time() - t
        simdata[i+1,2:] = currentPos
        print(f"{i} : {u}")
    
    # print(simdata)
    plotSimdata(simdata)

    # u = nmpc.solve(targetPos,currentPos,obstacles,cbf)
    print(u)
    print(nmpc.stateHorizon[-1,:])
