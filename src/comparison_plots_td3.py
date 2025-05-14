import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pickle

def plotSimdata(simdata,simdata2,simdata3,env):

    ob = env['obstacles']
    target = env['target_pos']
    # rewards = ep.rewards[:-1]
    # finalReward = ep.rewards[-1]
   
    fig = plt.figure(figsize=(10, 6))
    
    ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    ax3 = plt.subplot2grid((2, 3), (1, 2))

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
    t2 = np.arange(simdata2.shape[0]).reshape(-1, 1)
    t3 = np.arange(simdata3.shape[0]).reshape(-1, 1)
    mpct = simdata[:,5]*1000     # scaled to ms
    mpct2 = simdata2[:,5]*1000
    mpct3 = simdata3[:,5]*1000

    mpcave = np.average(simdata[:,5])
    mpcave2 = np.average(simdata2[:,5])
    mpcave3 = np.average(simdata3[:,5])
    print(mpcave,mpcave2,mpcave3)

    x = simdata[:,0]
    y = simdata[:,1]
    x2 = simdata2[:,0]
    y2 = simdata2[:,1]
    x3 = simdata3[:,0]
    y3 = simdata3[:,1]



    
    # th = simdata[:,2]
    v = simdata[:,3]
    w = simdata[:,4]
    # r = simdata[:,17] #simdata[:,7] 
    # s[0] = s[1]
    n = simdata[:,18]
    a = simdata[:,19]
    a2 = simdata2[:,19]
    a3 = simdata3[:,19]


    # XY position plot
    ax1.plot(x, y,          label="cbf-td3",    c='black',  lw = 0.8)      # Plots y versus x as a line
    ax1.scatter(x2, y2,s=2, label="cbf-0.01",   c='cyan',   alpha = 0.5)
    ax1.scatter(x3, y3,s=4, label="cbf-0.5",    c='green',  alpha=0.5)
    ax1.add_patch(Circle(simdata[-1,0:2], 0.55, color='black', alpha=0.9,))
    ax1.add_patch(Circle(target[0:2], 0.2, color='green', alpha=0.9))
    for i in range(ob.shape[0]):
        ax1.add_patch(Circle( ob[i,0:2], ob[i,2], color='red', alpha=0.7)) 
    ax1.legend(fontsize="small")
    
    # reward plot
    # ax3.plot(t,r, label="reward")
    
    # vehicle controls plot
    # ax2.plot(t,v, label="v (m/s)")
    # ax2.set_ylim(-0.1,1.1)
    # ax2b = ax2.twinx()
    # ax2b.plot(t,w, label=r'$\omega$ (rad/s)', color='orange')
    # ax2b.set_ylim(-1,1)
    

    # mpc time plot
    ax2.plot(t,mpct,    label=f"{round(mpcave*1000,1)} ms average td3",     c='black')
    ax2.plot(t2,mpct2,  label=f"{round(mpcave2*1000,1)} ms average 0.01",   c='cyan')
    ax2.plot(t3,mpct3,  label=f"{round(mpcave3*1000,1)} ms average 0.5",    c='green')
    ax2.set_ylim(10,100)
    ax2.legend(fontsize="small")

    # ax5.plot(t,n, label="NMPC-N")
    # # ax5.set_ylim(0, 100)
    # ax3b = ax3.twinx()
    # ax3b.plot(t, a, label="action", color='orange')
    # ax3b.set_ylim(0, 1)
    ax3.plot(t,a,   label="td3",         c='black')
    ax3.plot(t2,a2, label="cbf-0.01",    c='cyan')
    ax3.plot(t3,a3, label="cbf-0.5",     c='green')
    ax3.set_ylim(0,1)
    ax3.legend(fontsize="small")

    # Set axis limits for ax1
    lim = np.max(target[0:2])
    ax1.set_xlim(0, lim)
    ax1.set_ylim(-10, lim-10)

    # Set axis labels
    ax1.set_xlabel('X position (m)')
    ax1.set_ylabel('Y position (m)')
    ax1.set_title(f"Obstacle Radius: {ob[0,2]} m")
    # ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Solve Time (ms)')
    ax3.set_xlabel('Simulation Step')
    ax3.set_ylabel('CBF Value')
    # ax4.set_xlabel('Simulation Step')
    # ax4.set_ylabel('Linear Velocity')
    # ax4b.set_ylabel('Angular Velocity', color='orange')
    # ax5.set_xlabel('Simulation Step')
    # ax5.set_ylabel('NMPC Horizon')
    # ax5b.set_ylabel('CBF Action', color='orange')
    plt.tight_layout()

    # plt.figure(2)
    # plt.plot(t[1:],rewards)
    # plt.xlabel("Simulation Step")
    # plt.ylabel("Reward")

    plt.show()


if __name__ == "__main__":
    obs = 4.8
    # Load datafiles
    with open(f'test_data/data-rl-cbf1-4.8-0.0487.pkl', 'rb') as f: 
        simdata = pickle.load(f)

    with open(f'test_data/data-{obs}-cbf-0.01.pkl', 'rb') as f: 
        simdata2 = pickle.load(f)

    with open(f'test_data/data-{obs}-cbf-0.5.pkl', 'rb') as f: 
        simdata3 = pickle.load(f)

    with open(f'test_data/test-1_obs-{obs}.pkl', 'rb') as f: 
        map = pickle.load(f)

    plotSimdata(simdata,simdata2,simdata3,map)

