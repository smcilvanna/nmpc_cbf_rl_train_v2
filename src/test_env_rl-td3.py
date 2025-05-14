import numpy as np
from custom_env_horizon import MPCHorizonEnv
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import pickle
from stable_baselines3 import TD3
import tkinter as tk
from tkinter import filedialog

def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename()
    return file_path

def plotSimdata(simdata,env):

    ob = env['obstacles']
    target = env['target_pos']
    # rewards = ep.rewards[:-1]
    # finalReward = ep.rewards[-1]
   
    fig = plt.figure(figsize=(10, 6))
    
    ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2)    # ax1: left, spanning 2 rows and 2 columns  
    ax2 = plt.subplot2grid((2, 6), (0, 2), rowspan=1, colspan=2)    # ax2: 1 row, 2 columns top row right of ax1
    ax3 = plt.subplot2grid((2, 6), (1, 2), rowspan=1, colspan=2)    # ax3: 1 row, 2 columns bottom row right of ax1
    ax4 = plt.subplot2grid((2, 6), (0, 4), rowspan=1, colspan=2)    # ax4: 1 row, 2 columns right of ax2
    ax5 = plt.subplot2grid((2, 6), (1, 4), rowspan=1, colspan=2)    # ax4: 1 row, 2 columns right of ax3

    ax1.axis('square')

    # # Simdata Spec
    # log_row = env.current_pos.tolist()  # [0:3]     (3)
    # log_row.extend(info["u"].tolist())  # [3:5]     (2)         5         6            7           8                  9        10        11       12         13        14         15        16
    # log_row.extend(next_obs.tolist())   # [5:17]    (12)    [mpc_time, target_dist, target_sin, target_cos] + obs*(obs_dist, obs_sin, obs_cos  obs_rad) + (lin_vel ave_lin_vel, sin(yaw) cos(yaw))
    # log_row.extend([reward])            # [17]      (1)
    # log_row.extend([env.nmpc.currentN]) # [18]      (1)
    # log_row.extend(action.tolist())     # [19]      (1)     Size of obsstacle_attention

    # Position Plot
    t = np.arange(simdata.shape[0]).reshape(-1, 1)
    mpct = simdata[:,5]*1000     # scaled to ms
    x = simdata[:,0]
    y = simdata[:,1]
    th = simdata[:,2]
    v = simdata[:,3]
    w = simdata[:,4]
    r = simdata[:,17] #simdata[:,7] 
    # s[0] = s[1]
    n = simdata[:,18]
    a = simdata[:,19]

    # XY position plot
    ax1.scatter(x, y,s=1)      # Plots y versus x as a line
    ax1.add_patch(Circle(simdata[-1,0:2], 0.55, color='black', alpha=0.9, label="vehicle"))
    ax1.add_patch(Circle(target[0:2], 0.2, color='green', alpha=0.9))
    for i in range(ob.shape[0]):
        ax1.add_patch(Circle( ob[i,0:2], ob[i,2], color='red')) 

    # mpc time plot
    ax2.plot(t,mpct, label="mpc_time")
    
    # reward plot
    ax3.plot(t,r, label="reward")
    
    # vehicle controls plot
    ax4.plot(t,v, label="v (m/s)")
    ax4.set_ylim(-0.1,1.1)
    ax4b = ax4.twinx()
    ax4b.plot(t,w, label=r'$\omega$ (rad/s)', color='orange', alpha = 0.5)
    ax4b.set_ylim(-1,1)
    
    ax5.plot(t,n, label="NMPC-N")
    ax5.set_ylim(0, 100)
    ax5b = ax5.twinx()
    ax5b.plot(t, a, label="action", color='orange')
    ax5b.set_ylim(0, 1)

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
    ax3.set_ylabel('Reward')
    ax4.set_xlabel('Simulation Step')
    ax4.set_ylabel('Linear Velocity')
    ax4b.set_ylabel('Angular Velocity', color='orange')
    ax5.set_xlabel('Simulation Step')
    ax5.set_ylabel('NMPC Horizon')
    ax5b.set_ylabel('CBF Action', color='orange')
    plt.tight_layout()

    # plt.figure(2)
    # plt.plot(t[1:],rewards)
    # plt.xlabel("Simulation Step")
    # plt.ylabel("Reward")

    plt.show()

# def plotSimdataAnimated(ep,env):
    
#     # simdata = ep2simdata(ep)
#     ob = env['obstacles']
#     target = env['target_pos']
        
#     fig = plt.figure(figsize=(10, 6))
    
#     ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2)
#     ax2 = plt.subplot2grid((2, 6), (0, 2), rowspan=1, colspan=2)
#     ax3 = plt.subplot2grid((2, 6), (1, 2), rowspan=1, colspan=2)
#     ax4 = plt.subplot2grid((2, 6), (0, 4), rowspan=1, colspan=2)
#     ax5 = plt.subplot2grid((2, 6), (1, 4), rowspan=1, colspan=2)

#     # Set axis limits for ax1
#     lim = np.max(target[0:2])
#     ax1.set_xlim(0, lim)
#     ax1.set_ylim(0, lim)

#     # Set axis labels
#     ax1.set_xlabel('X position')
#     ax1.set_ylabel('Y position')
#     ax2.set_xlabel('Simulation Step')
#     ax2.set_ylabel('Solve Time (ms)')
#     ax3.set_xlabel('Simulation Step')
#     ax3.set_ylabel('Min Obs Separation (m)')
#     ax4.set_xlabel('Simulation Step')
#     ax4.set_ylabel('Velocity Controls')
#     ax5.set_xlabel('Simulation Step')
#     ax5.set_ylabel('Solver Horizon')

#     # Initialize plots with empty data that will be filled during animation
#     scatter = ax1.scatter([], [], s=10, color='blue')
#     vehicle_circle = Circle((0, 0), 0.55, color='black', alpha=0.9)
#     ax1.add_patch(vehicle_circle)
#     target_circle = Circle(target[0:2], 0.2, color='green', alpha=0.9)
#     ax1.add_patch(target_circle)
    
#     # Add obstacles
#     for obs in ob:
#         circle = Circle(obs[0:2], obs[2], color='red')
#         ax1.add_patch(circle)

#     # Initialize empty line plots
#     line_mpct, = ax2.plot([], [], label='mpc_time')
#     line_s, = ax3.plot([], [], label='min sep')
#     hlines = ax3.hlines(0, simdata[0,0], simdata[-1,0], colors='red')
#     line_v, = ax4.plot([], [], label='Linear (m/s)')
#     line_w, = ax4.plot([], [], label='Angular (rad/s)')
#     line_n, = ax5.plot([], [], label='Solver Horizon')

#     # Set the axis limits based on the full data range
#     ax2.set_xlim(simdata[0,0], simdata[-1,0])
#     ax3.set_xlim(simdata[0,0], simdata[-1,0])
#     ax4.set_xlim(simdata[0,0], simdata[-1,0])
#     ax4.set_xlim(simdata[0,0], simdata[-1,0])
#     ax5.set_xlim(simdata[0,0], simdata[-1,0])

#     ax2.set_ylim(0, np.max(simdata[:,1]*1000)*1.1)
#     ax3.set_ylim(0, np.max(simdata[:,7])*1.1)
#     ax4.set_ylim(np.min([simdata[:,5],simdata[:,6]])*1.1, np.max([simdata[:,5],simdata[:,6]])*1.1)
#     ax5.set_ylim(np.min(simdata[:,8])*1.1, np.max(simdata[:,8])*1.1)

#     # Add legends
#     ax2.legend()
#     ax3.legend()
#     ax4.legend()
#     ax4.legend()

#     # Define the update function for animation
#     def update(frame):
#         # Update scatter trail showing vehicle path
#         scatter.set_offsets(np.c_[simdata[:frame,2], simdata[:frame,3]])
#         # Update vehicle position
#         vehicle_circle.center = (simdata[frame,2], simdata[frame,3])

#         # Update time series plots
#         line_mpct.set_data(simdata[:frame,0], simdata[:frame,1]*1000)
#         line_s.set_data(simdata[:frame,0], simdata[:frame,7])
#         line_v.set_data(simdata[:frame,0], simdata[:frame,5])
#         line_w.set_data(simdata[:frame,0], simdata[:frame,6])
#         line_n.set_data(simdata[:frame,0], simdata[:frame,8])

#         return scatter, vehicle_circle, line_mpct, line_s, line_v, line_w, line_n

#     # Create animation
#     ani = FuncAnimation(fig, update, frames=len(simdata), interval=100, blit=True)

#     plt.tight_layout()
#     plt.show()
    
#     return ani  # Return animation object to prevent garbage collection


def normal_action_to_cbf(value,new_min=0.005,new_max=0.70):
    old_min = -1.0
    old_max = 1.0
    # new_min = 0.002             #0.01 #run2        #0.001 #run1
    # new_max = 0.70              #run2        #1.50 #run1
    scale = (new_max - new_min) / (old_max - old_min)       # Calculate the scaling factor
    real_value = float(new_min + (value - old_min) * scale) # Apply the transformation
    # print(f"Old Value: {value}, New Value: {real_value}")
    return real_value

def normalise_observation(value):
    qval = np.round(np.ceil(value/0.2)*0.2,1)
    dval = qval/0.2 -1
    return min(dval,24)



if __name__ == "__main__":
    # Manual test configuration
    MAX_STEPS = 500  # Reduce steps for easier debugging
    
    # Create wrapped environment
    env = MPCHorizonEnv(curriculum_level=1)
    obs, _ = env.reset()
    done = False
    step = 0
    last_action = None
    action_counter = 0
    
    # print(f"Initial observation: {obs[:4]}... (truncated)")
    log = []
    cnt = -10

    # Load Horizon Prediction Model
    # modelfile = 
    model = TD3.load("temp_td3/models/td3_run6_chkpt_5300_steps.zip")
    print(env.map["obstacles"][0:2])
    
    # Change main obstacle for plot
    main_rad = 5.0
    env.map["obstacles"][0,2] = main_rad
    env.map["obstacles"][1] = env.map["obstacles"][0]
    print(env.map["obstacles"][0:2])

    obstacles = env.map["obstacles"]
    # cbfs = []
    # for row in obstacles:
    #     rad = row[2]
    #     rad = normalise_observation(rad)
    #     ncbf, _ = model.predict(rad,deterministic=True)
    #     # ocbf = normal_action_to_cbf(ncbf)
    #     ocbf = (ncbf+1)/2
    #     # print(rad, ncbf, ocbf)
    #     cbfs.extend([ocbf])
    # cbfs = np.array(cbfs)
    # print(cbfs.shape)
    rad = normalise_observation(main_rad)
    ncbf, _ = model.predict(rad,deterministic=True)
    ocbf = np.float64((ncbf+1)/2)

    while not done and step < MAX_STEPS:
        # Use Model To Set Actions (cbf parameters)
        # action, _ = model.predict(obs, deterministic=True)#np.ones((env.obstacle_attention,))*(0.5 + cnt*0.045)
        action = ocbf
        print("cbf action " , action)
        if cnt == 10:
            cnt = -10
        else:
            cnt += 1 
        next_obs, reward, done, _, info = env.step(action)

        # Logging
        log_row = env.current_pos.tolist()  # [0:3]     (3)
        log_row.extend(info["u"].tolist())  # [3:5]     (2)
        log_row.extend(next_obs.tolist())   # [5:17]    (12)    [mpc_time, target_dist, target_sin, target_cos] + obs*(obs_dist, obs_sin, obs_cos  obs_rad) + (lin_vel ave_lin_vel, sin(yaw) cos(yaw))
        log_row.extend([reward])            # [17]      (1)
        log_row.extend([env.nmpc.currentN]) # [18]      (1)
        log_row.extend([action])     # [19]      (1)     Size of obsstacle_attention
        log.append(log_row)

        # Print step info
        print(f"\nStep {step+1}:")
        
        step += 1
        obs = next_obs
    
    # Plot run
    simdata = np.array(log)
    print(f"Total Reward : {np.sum(simdata[:,17])}")
    print(f"Rewards Before Terminal : {np.sum(simdata[:-1,17])}")
    plotSimdata(simdata,env.map)
