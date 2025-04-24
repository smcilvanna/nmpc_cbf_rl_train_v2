
# from generateCurriculumEnvironment import generate_curriculum_environment as genenv
# from generateCurriculumEnvironment import MapLimits
# from tkinter import filedialog
# import tkinter as tk
import numpy as np
from generateCurriculumEnvironment import genCurEnv_2 as genenv2
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import pickle

from nmpc_cbf import NMPC_CBF_MULTI_N
from episodeTracker import EpisodeTracker

np.set_printoptions(precision=3, suppress=True)


def plotSimdataAnimated(ep,env):
    
    simdata = ep2simdata(ep)
    ob = env['obstacles']
    target = env['target_pos']
        
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
    for obs in ob:
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

def ep2simdata(ep):
    # Observations Vector (93)
    #                        0     1     2         3        4         5       6    7
    #   state (8)       : [x-pos y-pos sin(yaw) cos(yaw)  x-tgt-n   y-tgt-n   v    w ]
    #                       8+o      9+o        10+o      11+o    (o =obsIndex*4)
    #   obsObv (80)     : [dist, sin(angle), cos(angle), radius]
    #                         88    89           90          91
    #   targetInfo (4)  : [ dist, sin(angle), cos(angle) progress]
    #                         92
    #   mpcTime  (1)    : [ mpcTime ]  
    #format epdata to simdata
    epdata = ep.all_observations
    acdata = ep.actions
    simdata = np.zeros((len(epdata),10))
    for i,row in enumerate(epdata):
        simdata[i,0] = i        # step number
        simdata[i,1] = row[92]  # mpc time
        simdata[i,2] = row[0]   # x position
        simdata[i,3] = row[1]   # y position
        simdata[i,4] = np.arcsin(row[2])    # yaw
        simdata[i,5] = row[6]   # v
        simdata[i,6] = row[7]   # w
        simdata[i,7] = 1e5      # min obstacle separation
        for o,_ in enumerate(obstacles):
            if row[8+o*4] < simdata[i,7]:
                simdata[i,7] = row[8+o*4]
        simdata[i,8] = acdata[i][-1] # mpc horizon length
    
    return simdata

def plotSimdata(ep,env):

    simdata = ep2simdata(ep)
    ob = env['obstacles']
    target = env['target_pos']
    rewards = ep.rewards[:-1]
    finalReward = ep.rewards[-1]
   
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
    for i in range(ob.shape[0]):
        ax1.add_patch(Circle( ob[i,0:2], ob[i,2], color='red')) 

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

    plt.figure(2)
    plt.plot(t[1:],rewards)
    plt.xlabel("Simulation Step")
    plt.ylabel("Reward")

    plt.show()

def simulateStep(startState,cbf):
    t = time.time()
    u = nmpc.solve(startState,cbf)
    currentPos = nmpc.stateHorizon[0,:]
    mpcTime = time.time() - t
    return currentPos, u, mpcTime         

def checkCollision(vehiclePos, obs):
    cen_sep = np.linalg.norm(vehiclePos[0:2] - obs[0:2])
    safe_sep = cen_sep - 0.55 - obs[2]
    collision = safe_sep <= 0.0 
    return collision, safe_sep

def calculate_reward(ep,isdone):
    # Observations list (93)
    #                        0     1     2         3        4         5       6    7
    #   state (8)       : [x-pos y-pos sin(yaw) cos(yaw)  x-tgt-n   y-tgt-n   v    w ]
    #                       8+o      9+o        10+o      11+o    (o =obsIndex*4)
    #   obsObv (80)     : [dist, sin(angle), cos(angle), radius]
    #                         88    89           90          91
    #   targetInfo (4)  : [ dist, sin(angle), cos(angle) progress]
    #                         92
    #   mpcTime  (1)    : [ mpcTime ] 
    observations = ep.latest_observation    
    
    # --- Every Step Components ---
    # Velocity Maintenance (Gaussian around 1 m/s)
    velocity = observations[6]
    rv = 1.0 * np.exp(-2 * (velocity - 1.0)**2)         # >>> Reward interval [0 1]
    
    # Progress : At 1m/s max velocity maximum progress would be 0.1m per 0.1s timestep
    # negative progress is not penalised
    rp=0
    if len(ep.past_observations) >= 1:
        prev_dist = ep.past_observations[-1][88]
        curr_dist = ep.latest_observation[88]
        rp = np.tanh(10*(max(prev_dist - curr_dist,0))) ## >>> Reward interval [0 1]
    
    # Action Smoothness Penalty
    # CBF actions are the first elements in the action array, up to the number of obstacles
    # N action is always the last element in the action array
    # All actions are normalised in range [0 1]
    # Maximum reward for minimal change in action, decaying to zero for large changes
    if len(ep.actions) >= 2:
        current_actions = np.array(ep.actions[-1])
        prev_actions = np.array(ep.actions[-2])
        # cbf_diff = np.linalg.norm(np.array(current_actions[0:-1]) - np.array(prev_actions[0:-1])) / 2.0  # this is wrong , cbf actions will be in [0 1] range
        # n_diff = np.linalg.norm(current_actions[-1] - prev_actions[-1]) / 2.0 # this is wrong , N action will be in [0 1] range
        # cbf_diff = np.abs(np.array(current_actions[0:-1]) - np.array(prev_actions[0:-1]))
        cbf_diff = np.linalg.norm(current_actions[0:-1] - prev_actions[0:-1])  # L2 norm                        
        ra = np.exp(-5 * cbf_diff**2)                                       # Reward interval [0 1]
        # not punishing mpc changes as this is reflected in mpc time
        # n_ra = np.exp(-7 * (abs(current_actions[-1]-prev_actions[-1]))**2)  # Reward interval [0 1]
        # ra = (0.8*cbf_ra + 0.2*n_ra)                                        # Total action reward interval [0 1]
    else:
        ra = 1
    # Solver Time, want to encourage solver time < 100ms
    mpctime = observations[92]
    tt = 0.090  
    if mpctime < tt:
        rt = 1.0                                    # mpc time below threshold (50ms) full reward
    else:                                           # anything above 50ms decayed reward, almost zero beyond 150ms
        rt = np.exp(-3 * ((mpctime - tt)/0.1)**2)   # Reward interval [0 1]

    # reward intervals    [0 1]     [0 1]     [0 1]     [0 1]
    
    reward      =        0.2*rv   +    0.2*rp    +   0.2*ra    +   0.4*rt   # total step reward
    
    reward = reward/4   # Step reward interval [0 1]
    
    #                 0       1         2         3     
    # isdone list : [done, atTarget, collision, tooSlow] 
    
    # --- Terminal/Episodic Bonuses ---
    if isdone[1]:
        reward += 200 
        reward += 20 * ep.epPassGates  # Bonus for navigating close to obstacle
    
    # Collision Penalty (Terminal)
    if isdone[2]:
        reward -= 250

    # Deadlock Penalty (Terminal)
    if isdone[3]:
        reward -= 150
    
    return reward


def getStepObservations(currentState,u,mpcTime,env,normalise=False):
    # Observations Vector (93)
    #                        0     1     2         3        4         5       6    7
    #   state (8)       : [x-pos y-pos sin(yaw) cos(yaw)  x-tgt-n   y-tgt-n   v    w ]
    #                       8+o      9+o        10+o      11+o    (o =obsIndex*4)
    #   obsObv (80)     : [dist, sin(angle), cos(angle), radius]
    #                         88    89           90          91
    #   targetInfo (4)  : [ dist, sin(angle), cos(angle) progress]
    #                         92
    #   mpcTime  (1)    : [ mpcTime ]  
    targetPos = env['target_pos']
    obstacles = env['obstacles']
    state = currentState.tolist()
    state.extend([0.0, np.sin(state[2]), np.cos(state[2])])
    state.extend(u.tolist())
    # state[2] = np.max([0, np.min([1,(targetPos[0]-state[0])/np.max([0.0001,targetPos[0]])])]) # scaled to target normalised x
    state[2] = max(0, min(1, (targetPos[0] - state[0]) / max(0.0001, targetPos[0])))
    # state[3] = np.max([0, np.min([1,(targetPos[1]-state[1])/np.max([0.0001,targetPos[1]])])]) # scaled to target normalised y
    state[3] = max(0, min(1, (targetPos[1] - state[1]) / max(0.0001, targetPos[1])))

    obsObsv =[]
    for o in obstacles:
        obs_metrics = obstacle_metrics(state[0:2], o)
        obsObsv.extend(obs_metrics.tolist())
        # obsObsv = np.append(obsObsv,obstacle_metrics(state[0:2], o))

    targetInfo = obstacle_metrics(state[0:2], np.append(targetPos, 0.1)).tolist()
    # targetInfo = obstacle_metrics(state[0:2],np.append(targetPos, 0.1))                     # target area [distance sin() cos() radii]
    targetInfo[3] = max(0.00, env["startDist"] - targetInfo[0]) / env["startDist"]
    # targetInfo[3] = np.max((0.00, env["startDist"] - targetInfo[0] )) / env["startDist"]    # replace targetInfo radii with progress %
    
    observations = []
    observations.extend(state)
    observations.extend(obsObsv)
    observations.extend(targetInfo)
    observations.append(mpcTime)

    if normalise:
        return normalise_observations(observations)
    else:
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
    # Observations Vector (93)
    #                        0     1     2         3        4         5       6    7
    #   state (8)       : [x-pos y-pos sin(yaw) cos(yaw)  x-tgt-n   y-tgt-n   v    w ]
    #                       8+o      9+o        10+o      11+o    (o =obsIndex*4)
    #   obsObv (80)     : [dist, sin(angle), cos(angle), radius]
    #                         88    89           90          91
    #   targetInfo (4)  : [ dist, sin(angle), cos(angle) progress]
    #                         92
    #   mpcTime  (1)    : [ mpcTime ]  
    # obs = obs.flatten()

    full_labels = [
        "Current X", "Current Y", "sin(yaw)", "cos(yaw)", "X/x-target", "Y/y-target", "Velocity(v)", "Steering(w)"
    ]
    for i in range(nmpc.nObs):
        full_labels += [
            f"Obstacle {i+1} Clearance", f"Obstacle {i+1} sin(θ)",
            f"Obstacle {i+1} cos(θ)", f"Obstacle {i+1} radius"
        ]
    full_labels += [
        "Target Distance", "Target sin(θ)", "Target cos(θ)", "Target Progress %",
        "MPC Time"
    ]

    for l, v in zip(full_labels, obs):
        print(f"{l}: {v:.4f}")

def obstacle_metrics(state, obstacle):
    dx, dy = obstacle[0] - state[0], obstacle[1] - state[1]
    dist = np.hypot(dx, dy) - (nmpc.vehRad + obstacle[2])
    angle = np.arctan2(dy, dx)
    return np.array([dist, np.sin(angle), np.cos(angle), obstacle[2]])

def episodeTermination(ep):
    done = atTarget = collision = tooSlow = False
    # Check if episode is complete
    # 1: Check for at target:
    if ep.latest_observation[88] < 0.1:
        atTarget = True
        print("[Episode End] At Target")
    # 2: Check for collision
    for i in range(19):
        if ep.latest_observation[8+i*4] <= 0.00 :
            collision = True
            print(f"[Episode End] Obstacle {i} CRASH!")
            break

    # Check for Stuck
    if not atTarget and len(ep.past_observations) == 5:
        if np.mean([sublist[6] for sublist in ep.past_observations]) < 0.1:
            tooSlow = True
            print(f"[Episode End] Too Slow!")
    # Check for any termination event
    done = atTarget or collision or tooSlow
    return [done, atTarget, collision, tooSlow]

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
        
        # Switch horizon
        print(cnt)
        cnt = cnt+1
        if cnt % 20 == 0:
            # nmpc.adjustHorizon(np.random.uniform(0,1))
            # cbf = np.random.uniform(0,1, size=(1,nmpc.nObs))
            continue
        if cnt > maxSimSteps:
            print("EPISODE TIMEOUT")
            ep.done = True
    

    # plotSimdata(ep,env)
    # ani = plotSimdataAnimated(ep,env)
    # startPos = simdata[-1,2:5]
    # simdata = simulateStep(10,startPos,obstacles)
    # plotSimdata(simdata,targetPos)



    # # terminal state checks
    
    # # how many path targets hit
    # targetPaths = env["pass_targets"]
    # tpCnt = 0
    # for pgt in targetPaths.reshape(-1, targetPaths.shape[-1]):
    #     for st in simdata[:,0:2]:
    #         hitPath, _ = checkCollision(st,np.append(pgt,0.55))
    #         if hitPath:
    #             tpCnt += 1
    #             break

    # # collision
    # collide = np.min(simdata[:,7]) <= 0.00
