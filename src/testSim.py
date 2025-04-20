
import numpy as np
from generateCurriculumEnvironment import generate_curriculum_environment as genenv
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from nmpc_cbf import NMPC_CBF_MULTI_N

np.set_printoptions(precision=3, suppress=True)

def plotSimdata(simdata,target):
    fig = plt.figure(figsize=(10, 6))
    
    ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2)    # ax1: left, spanning 2 rows and 2 columns  
    ax2 = plt.subplot2grid((2, 6), (0, 2), rowspan=1, colspan=2)    # ax2: 1 row, 2 columns top row right of ax1
    ax3 = plt.subplot2grid((2, 6), (1, 2), rowspan=1, colspan=2)    # ax3: 1 row, 2 columns bottom row right of ax1
    ax4 = plt.subplot2grid((2, 6), (0, 4), rowspan=1, colspan=2)    # ax4: 1 row, 2 columns right of ax2
    ax5 = plt.subplot2grid((2, 6), (1, 4), rowspan=1, colspan=2)    # ax4: 1 row, 2 columns right of ax3

    # Position Plot
    t = simdata[:,0]
    mpct = simdata[:,1]*1000
    x = simdata[:,2]
    y = simdata[:,3]
    th = simdata[:,4]
    v = simdata[:,5]
    w = simdata[:,6]
    s = simdata[:,7] 
    s[0] = s[1]

    ax1.scatter(x, y,s=1)      # Plots y versus x as a line
    ax1.add_patch(Circle(simdata[-1,2:5], 0.55, color='black', alpha=0.9, label="vehicle"))
    ax1.add_patch(Circle(target[0:2], 0.2, color='green', alpha=0.9))
    for i in range(obstacles.shape[0]):
        ax1.add_patch(Circle( obstacles[i,0:2], obstacles[i,2], color='red')) 

    ax2.plot(t,mpct, label="mpc_time")
    ax3.plot(t,s, label="min sep")
    ax3.hlines(0,t[0],t[-1], colors='red')
    ax4.plot(t,v, label="Linear (m/s)")
    ax5.plot(t,w, label="Angular (rad/s)")

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
    ax3.set_ylabel('Min Obs Separation (m)')
    ax4.set_xlabel('Simulation Step')
    ax4.set_ylabel('v (m/s)')
    ax5.set_xlabel('Simulation Step')
    ax5.set_ylabel(r'$\omega$ (rad/s)')

    plt.tight_layout()
    plt.show()

def simulateStep(stepTime, startState, obstacles ,cbf ):
    simSteps = int(stepTime / 0.1)          # hardcoded timestep dt = 0.1
    # print(simSteps)
    simdata = np.zeros(( simSteps + 1, 8))  # [stepIndex solveTime x y yaw v w]
    simdata[0,2:5] = startState
    currentPos = startState
    for i in range(1,simSteps+1):
        t = time.time()
        u = nmpc.solve(targetPos,currentPos,obstacles,cbf)
        currentPos = nmpc.stateHorizon[1,:]
        simdata[i,0] = i
        simdata[i,1] = time.time() - t
        simdata[i,2:5] = currentPos
        simdata[i,5:7] = u
        # print(f"{i} : {u}")
        collision = False
        sep = 1000.0
        for obs in obstacles:
            c, s = checkCollision(currentPos,obs)
            collision = collision and c
            if s < sep:
                sep = s
        simdata[i,7] = sep
        if collision:
            print("CRASH!")
            break
    return simdata         

def checkCollision(vehiclePos, obstacle):
    cen_sep = np.linalg.norm(vehiclePos[0:2] - obstacle[0:2])
    safe_sep = cen_sep - 0.55 - obstacle[2]
    collision = safe_sep <= 0.0 
    return collision, safe_sep

if __name__ == "__main__":
    print("[START]")
    env = genenv(3, gen_fig=False)
    # plt.show()
    # print(env["obstacles"])
    # print(env)
    # exit()

    nmpc = NMPC_CBF_MULTI_N(0.1, [10, 20, 30, 40, 50], 20)
    print("NMPC_CBF_MULTI_N class initialized successfully.")
    nmpc.solversIdx = 2
    nmpc.currentN = nmpc.nVals[nmpc.solversIdx]
    obstacles = env['obstacles']
    targetPos = env['target_pos']
    # print(targetPos)
    # print(np.rad2deg(targetPos[2]))

    startPos = np.array([0,0,targetPos[2]])
    # cbf = np.array([ 1,1,1,1,1,1])*1

    epStepTime = 5
    epMaxTime = 15
    # epMaxSteps = epMaxTime / nmpc.dt
    # epdata = np.zeros(( epMaxSteps + 1, 8))
    runtime = 0
    while runtime < epMaxTime:    
        cbf = np.random.randint(1, 101, size=(1, 20))
        simdata = simulateStep(epStepTime,startPos,obstacles, cbf)
        startPos = simdata[-1,2:5]
        runtime += epStepTime
        if runtime == epStepTime:
            epdata = simdata
        else:
            epdata = np.vstack([epdata,simdata[1:,:]])
        print(">> ",end=None)

    print(epdata)
    print(epdata.shape)
    for i in range(301):
        epdata[i,0]=i

    plotSimdata(epdata,targetPos)

    # startPos = simdata[-1,2:5]
    # simdata = simulateStep(10,startPos,obstacles)
    # plotSimdata(simdata,targetPos)


