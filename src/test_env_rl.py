import numpy as np
from custom_env_horizon import MPCHorizonEnv, ActionPersistenceWrapper
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import pickle
from stable_baselines3 import PPO


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

    # Position Plot
    t = np.arange(simdata.shape[0]).reshape(-1, 1)
    mpct = simdata[:,5]*1000
    x = simdata[:,0]
    y = simdata[:,1]
    th = simdata[:,2]
    v = simdata[:,3]
    w = simdata[:,4]
    s = simdata[:,-2] #simdata[:,7] 
    # s[0] = s[1]
    n = simdata[:,-1]

    ax1.scatter(x, y,s=1)      # Plots y versus x as a line
    ax1.add_patch(Circle(simdata[-1,0:2], 0.55, color='black', alpha=0.9, label="vehicle"))
    ax1.add_patch(Circle(target[0:2], 0.2, color='green', alpha=0.9))
    for i in range(ob.shape[0]):
        ax1.add_patch(Circle( ob[i,0:2], ob[i,2], color='red')) 

    ax2.plot(t,mpct, label="mpc_time")
    ax3.plot(t,s, label="reward")
    # ax3.hlines(0,t[0],t[-1], colors='red')
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
    ax3.set_ylabel('Reward')
    ax4.set_xlabel('Simulation Step')
    ax4.set_ylabel('Velocity Controls')
    ax5.set_xlabel('Simulation Step')
    ax5.set_ylabel('Solver Horizon')
    plt.tight_layout()

    # plt.figure(2)
    # plt.plot(t[1:],rewards)
    # plt.xlabel("Simulation Step")
    # plt.ylabel("Reward")

    plt.show()

def plotSimdataAnimated(ep,env):
    
    # simdata = ep2simdata(ep)
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


if __name__ == "__main__":
    # Manual test configuration
    MAX_STEPS = 1000  # Reduce steps for easier debugging
    PERSIST_STEPS = 10  # Test action persistence interval
    
    # Create wrapped environment
    env = ActionPersistenceWrapper(MPCHorizonEnv(curriculum_level=2), persist_steps=PERSIST_STEPS)
    with open('./env-1-1.pkl', 'rb') as f: 
        map = pickle.load(f)

    obs, _ = env.reset(map=map)
    done = False
    step = 0
    last_action = None
    action_counter = 0
    
    # Load Horizon Prediction Model
    model = PPO.load("../../train_data/horizon_only/ppo_mpc_horizon_ks_1-3d_complex")

    # print(f"Initial observation: {obs[:4]}... (truncated)")
    log = []
    while not done and step < MAX_STEPS:
        # Take random action (will only be applied every `PERSIST_STEPS` steps)
        action, _ = model.predict(obs) #env.action_space.sample()
        next_obs, reward, done, _, info = env.step(action)

        # Logging for plots
        log_row = env.env.current_pos.tolist()
        log_row.extend(info["u"].tolist())
        log_row.extend(next_obs.tolist())
        log_row.extend([reward])
        log_row.extend([env.env.nmpc.currentN])
        log.append(log_row)

        # Print step info
        print(f"\nStep {step+1}:")
        
        step += 1
        obs = next_obs

# Plot run
simdata = np.array(log)
plotSimdata(simdata,env.env.map)
