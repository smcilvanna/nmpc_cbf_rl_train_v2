
import numpy as np
from generateCurriculumEnvironment import generate_curriculum_environment as genenv
from generateCurriculumEnvironment import genCurEnv_2 as genenv2
from generateCurriculumEnvironment import MapLimits
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from nmpc_cbf import NMPC_CBF_MULTI_N
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import filedialog
import pickle

np.set_printoptions(precision=3, suppress=True)


def plotSimdataAnimated(simdata, target, obstacles):
    fig = plt.figure(figsize=(10, 6))
    
    ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), rowspan=1, colspan=2)
    ax3 = plt.subplot2grid((2, 6), (1, 2), rowspan=1, colspan=2)
    ax4 = plt.subplot2grid((2, 6), (0, 4), rowspan=1, colspan=2)
    ax5 = plt.subplot2grid((2, 6), (1, 4), rowspan=1, colspan=2)

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
    ax4.set_ylabel('Velocity Controls')
    ax5.set_xlabel('Simulation Step')
    ax5.set_ylabel('Solver Horizon')

    # Initialize plots with empty data that will be filled during animation
    scatter = ax1.scatter([], [], s=10, color='blue')
    vehicle_circle = Circle((0, 0), 0.55, color='black', alpha=0.9)
    ax1.add_patch(vehicle_circle)
    target_circle = Circle(target[0:2], 0.2, color='green', alpha=0.9)
    ax1.add_patch(target_circle)
    
    # Add obstacles
    for obs in obstacles:
        circle = Circle(obs[0:2], obs[2], color='red')
        ax1.add_patch(circle)

    # Initialize empty line plots
    line_mpct, = ax2.plot([], [], label='mpc_time')
    line_s, = ax3.plot([], [], label='min sep')
    hlines = ax3.hlines(0, simdata[0,0], simdata[-1,0], colors='red')
    line_v, = ax4.plot([], [], label='Linear (m/s)')
    line_w, = ax4.plot([], [], label='Angular (rad/s)')
    line_n, = ax5.plot([], [], label='Solver Horizon')

    # Set the axis limits based on the full data range
    ax2.set_xlim(simdata[0,0], simdata[-1,0])
    ax3.set_xlim(simdata[0,0], simdata[-1,0])
    ax4.set_xlim(simdata[0,0], simdata[-1,0])
    ax4.set_xlim(simdata[0,0], simdata[-1,0])
    ax5.set_xlim(simdata[0,0], simdata[-1,0])

    ax2.set_ylim(0, np.max(simdata[:,1]*1000)*1.1)
    ax3.set_ylim(0, np.max(simdata[:,7])*1.1)
    ax4.set_ylim(np.min([simdata[:,5],simdata[:,6]])*1.1, np.max([simdata[:,5],simdata[:,6]])*1.1)
    ax5.set_ylim(np.min(simdata[:,8])*1.1, np.max(simdata[:,8])*1.1)

    # Add legends
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax4.legend()

    # Define the update function for animation
    def update(frame):
        # Update scatter trail showing vehicle path
        scatter.set_offsets(np.c_[simdata[:frame,2], simdata[:frame,3]])
        # Update vehicle position
        vehicle_circle.center = (simdata[frame,2], simdata[frame,3])

        # Update time series plots
        line_mpct.set_data(simdata[:frame,0], simdata[:frame,1]*1000)
        line_s.set_data(simdata[:frame,0], simdata[:frame,7])
        line_v.set_data(simdata[:frame,0], simdata[:frame,5])
        line_w.set_data(simdata[:frame,0], simdata[:frame,6])
        line_n.set_data(simdata[:frame,0], simdata[:frame,8])

        return scatter, vehicle_circle, line_mpct, line_s, line_v, line_w, line_n

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(simdata), interval=100, blit=True)

    plt.tight_layout()
    plt.show()
    
    return ani  # Return animation object to prevent garbage collection


def plotSimdata(simdata,target):
    fig = plt.figure(figsize=(10, 6))
    
    ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2)    # ax1: left, spanning 2 rows and 2 columns  
    ax2 = plt.subplot2grid((2, 6), (0, 2), rowspan=1, colspan=2)    # ax2: 1 row, 2 columns top row right of ax1
    ax3 = plt.subplot2grid((2, 6), (1, 2), rowspan=1, colspan=2)    # ax3: 1 row, 2 columns bottom row right of ax1
    ax4 = plt.subplot2grid((2, 6), (0, 4), rowspan=1, colspan=2)    # ax4: 1 row, 2 columns right of ax2
    ax5 = plt.subplot2grid((2, 6), (1, 4), rowspan=1, colspan=2)    # ax4: 1 row, 2 columns right of ax3

    ax1.axis('square')

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
    n = simdata[:,8]

    ax1.scatter(x, y,s=1)      # Plots y versus x as a line
    ax1.add_patch(Circle(simdata[-1,2:5], 0.55, color='black', alpha=0.9, label="vehicle"))
    ax1.add_patch(Circle(target[0:2], 0.2, color='green', alpha=0.9))
    for i in range(obstacles.shape[0]):
        ax1.add_patch(Circle( obstacles[i,0:2], obstacles[i,2], color='red')) 

    ax2.plot(t,mpct, label="mpc_time")
    ax3.plot(t,s, label="min sep")
    ax3.hlines(0,t[0],t[-1], colors='red')
    ax4.plot(t,v, label="v (m/s)")
    ax4.plot(t,w, label=r'$\omega$ (rad/s)')
    ax5.plot(t,n, label="NMPC-N")

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
    ax4.set_ylabel('Velocity Controls')
    ax5.set_xlabel('Simulation Step')
    ax5.set_ylabel('Solver Horizon')

    plt.tight_layout()
    plt.show()

def simulateStep(stepTime, startState, obstacles ,cbf ):
    simSteps = int(stepTime / 0.1)          # hardcoded timestep dt = 0.1
    simdata = np.zeros(( simSteps + 1, 10))     # [stepIndex solveTime x y yaw v w sep N  _ ]
    simdata[0,2:5] = startState                 #   0           1      2 3  4  5 6  7  8  9
    currentPos = startState
    for i in range(1,simSteps+1):
        t = time.time()
        u = nmpc.solve(targetPos,currentPos,obstacles,cbf)
        currentPos = nmpc.stateHorizon[1,:]
        simdata[i,0] = i
        simdata[i,1] = time.time() - t
        simdata[i,2:5] = currentPos
        simdata[i,5:7] = u
        collision = False
        sep = 1000.0
        for obs in obstacles:
            c, s = checkCollision(currentPos,obs)
            collision = collision and c
            if s < sep:
                sep = s
        simdata[i,7] = sep
        simdata[i,8] = nmpc.currentN
        if collision:
            print("CRASH!")
            break
    return simdata         

def checkCollision(vehiclePos, obstacle):
    cen_sep = np.linalg.norm(vehiclePos[0:2] - obstacle[0:2])
    safe_sep = cen_sep - 0.55 - obstacle[2]
    collision = safe_sep <= 0.0 
    return collision, safe_sep

def getStepObservations(simdata,obstacles,env):

    state = simdata[-1,2:5]                                                             
    state = np.append(state, [0.0, np.sin(state[2]), np.cos(state[2])])                      
    state[2] = np.max([0, np.min([1,(targetPos[0]-state[0])/np.max([0.0001,targetPos[0]])])]) # scaled to target normalised x
    state[3] = np.max([0, np.min([1,(targetPos[1]-state[1])/np.max([0.0001,targetPos[1]])])]) # scaled to target normalised y
    avev  = np.average(simdata[:,5])
    maxv  = np.max(simdata[:,5])
    avew  = np.average(abs(simdata[:,6]))
    maxw  = np.max(abs(simdata[:,6]))
    ave_mpct = np.average(simdata[:,1])
    max_mpct = np.max(simdata[:,1])
    obsObsv =np.empty((0,4))
    for o in obstacles:
        obsObsv = np.append(obsObsv,obstacle_metrics(state[0:2], o))

    # terminal state checks
    
    # how many path targets hit
    targetPaths = env["pass_targets"]
    tpCnt = 0
    for pgt in targetPaths.reshape(-1, targetPaths.shape[-1]):
        for st in simdata[:,0:2]:
            hitPath, _ = checkCollision(st,np.append(pgt,0.55))
            if hitPath:
                tpCnt += 1
                break

    # collision
    collide = np.min(simdata[:,7]) <= 0.00

    # at target
    tgt = np.append(targetPos, 0.1)
    targetInfo = obstacle_metrics(state[0:2],tgt)
    targetDist = startDist - targetInfo[0]
    targetProgress = np.max((0.00,targetDist)) / startDist

    observations = np.hstack([  state.reshape((1,-1)),      # 4
                                obsObsv.reshape((1,-1)),    # 20*3 = 80
                                targetInfo[0:3].reshape((1,-1)), # 4 
                                np.array([[avev, maxv, avew, maxw, ave_mpct, max_mpct, tpCnt, targetProgress]])  ])
    print_observations(observations)
    print(observations.shape)
    normalised_observations = normalise_observations(observations)
    print_observations(normalised_observations)
    exit()
    return observations

def normalise_observations(obs):
    obs = obs.flatten()
    norm = obs.copy()

    # Normalisation assumptions
    # Position: [-10, 10] m → [-1, 1]
    norm[0:2] = np.clip(obs[0:2] / 10.0, -1.0, 1.0)
    
    # Normalised x y & Heading: sin(θ), cos(θ): already in [0 1] [-1, 1]
    norm[2:6] = obs[2:6]
    i = 6
    
    for _ in range(nmpc.nObs): 
        norm[i] = np.clip(obs[i] / 10.0, 0.0, 1.0)     # clearance
        norm[i+1] = obs[i+1]                           # sin(θ)
        norm[i+2] = obs[i+2]                           # cos(θ)
        norm[i+3] = np.clip(obs[i+3] / 10.0, 0.0, 1.0)
        i += 4

    # Target clearance (i), sin(θ), cos(θ)
    norm[i]   = np.clip(obs[i] / 10.0, 0.0, 1.0)
    norm[i+1] = obs[i+1]
    norm[i+2] = obs[i+2]
    i += 3

    # Motion metrics
    norm[i]   = np.clip(obs[i]   / 2.0,  0.0, 1.0)   # ave speed
    norm[i+1] = np.clip(obs[i+1] / 2.0,  0.0, 1.0)   # max speed
    norm[i+2] = np.clip(obs[i+2] / 2.0,  0.0, 1.0)   # ave angular
    norm[i+3] = np.clip(obs[i+3] / 2.0,  0.0, 1.0)   # max angular
    norm[i+4] = np.clip(obs[i+4] / 0.1,  0.0, 1.0)   # ave MPC t
    norm[i+5] = np.clip(obs[i+5] / 0.1,  0.0, 1.0)   # max MPC t

    # Targets hit: assume max 10 → [0, 1]
    norm[i+6] = np.clip(obs[i+6] / 10.0, 0.0, 1.0)

    # Target progress: already [0, 1]
    norm[i+7] = obs[i+7]

    return norm



def print_observations(obs):
    obs = obs.flatten()
    # n_fixed = 4
    # n_metrics = 4
    # n_target = 3
    # n_perf = 8

    full_labels = [
        "Current X", "Current Y", "sin(yaw)", "cos(yaw)", "X/x-target", "Y/y-target"
    ]
    for i in range(nmpc.nObs):
        full_labels += [
            f"Obstacle {i+1} Clearance", f"Obstacle {i+1} sin(θ)",
            f"Obstacle {i+1} cos(θ)", f"Obstacle {i+1} radius"
        ]
    full_labels += [
        "Target Clearance", "Target sin(θ)", "Target cos(θ)",
        "Average Speed", "Max Speed", "Average Angular Rate", "Max Angular Rate",
        "Average MPC Time", "Max MPC Time", "Targets Hit", "Target Progress"
    ]

    for l, v in zip(full_labels, obs):
        print(f"{l}: {v:.4f}")

def obstacle_metrics(state, obstacle):
    dx, dy = obstacle[0] - state[0], obstacle[1] - state[1]
    dist = np.hypot(dx, dy) - (nmpc.vehRad + obstacle[2])
    angle = np.arctan2(dy, dx)
    return np.array([dist, np.sin(angle), np.cos(angle), obstacle[2]])

if __name__ == "__main__":
    print("[START]")
    random_env = True
    if random_env:
        # env = genenv(2, gen_fig=True)
        # plt.show()
        # input("[ENTER] to begin")
        env = genenv2(curriculum_level=1,gen_fig=True)
        plt.show()
    else:
        file_path = './env4-1.pkl'
        with open(file_path, 'rb') as f:
            env = pickle.load(f)
    # exit()
    Nvalues = [10 , 30, 60 ,100] #np.arange(10,110,10)#[10, 20, 30, 40, 50]
    nmpc = NMPC_CBF_MULTI_N(0.1, Nvalues, 20)
    print("NMPC_CBF_MULTI_N class initialized successfully.")
    nmpc.solversIdx = np.random.randint(0,len(Nvalues)) # random start solver
    nmpc.currentN = nmpc.nVals[nmpc.solversIdx]
    obstacles = env['obstacles']
    # obstacles[:,1] -= 2.0
    targetPos = env['target_pos']
    startPos = np.array([0,0,targetPos[2]])
    startDist = np.linalg.norm(targetPos)
    epStepTime = 2
    epMaxTime = 15
    runtime = 0
    targetArea = np.append(targetPos,0.05)
    while runtime < epMaxTime:
        cbf = np.tile(1000,nmpc.nObs)#np.random.randint(1, 1000, size=(1, 20))/100 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CBF VALUES
        simdata = simulateStep(epStepTime,startPos,obstacles, cbf)
        
        # get observations for next step
        x = getStepObservations(simdata,obstacles,env)
        
        exit()
        
        startPos = simdata[-1,2:5]
        runtime += epStepTime
        if runtime == epStepTime:
            epdata = simdata
        else:
            epdata = np.vstack([epdata,simdata[1:,:]])
        print(">>")
        fin, _ = checkCollision(startPos,targetArea)
        if fin:
            print("At Target")
            break
        else:
            nmpc.adjustHorizon(np.random.randint(0,len(Nvalues)) ) 

    epdata[:,0] = np.arange(epdata.shape[0])

    # plotSimdata(epdata,targetPos)
    ani = plotSimdataAnimated(epdata, targetPos, obstacles)
    # startPos = simdata[-1,2:5]
    # simdata = simulateStep(10,startPos,obstacles)
    # plotSimdata(simdata,targetPos)


