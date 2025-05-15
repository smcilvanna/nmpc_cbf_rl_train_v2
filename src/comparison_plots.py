import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pickle

def plotSimdata(simdata,simdata2,simdata3,env):
    plot3 = True
    if simdata3 == None:
        plot3 = False

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
    t =  np.arange( simdata.shape[0]).reshape(-1, 1)
    t2 = np.arange(simdata2.shape[0]).reshape(-1, 1)
    mpct = simdata[:,5]*1000     # scaled to ms
    mpct2 = simdata2[:,5]*1000

    mpcave = np.average(simdata[:,5])
    mpcave2 = np.average(simdata2[:,5])
    # print(mpcave,mpcave2,mpcave3)

    x = simdata[:,0]
    y = simdata[:,1]
    x2 = simdata2[:,0]
    y2 = simdata2[:,1]

    # th = simdata[:,2]
    v = simdata[:,3]
    w = simdata[:,4]
    # r = simdata[:,17] #simdata[:,7] 
    # s[0] = s[1]
    n = simdata[:,18]
    a = simdata[:,-1]
    a2 = simdata2[:,-1]

    if plot3:
        a3 = simdata3[:,-1]
        x3 = simdata3[:,0]
        y3 = simdata3[:,1]
        mpcave3 = np.average(simdata3[:,5])
        mpct3 = simdata3[:,5]*1000
        t3 = np.arange(simdata3.shape[0]).reshape(-1, 1)

    c = ['blue', 'cyan', 'black']

    # XY position plot
    ax1.plot(x, y,      label="ppo",    c=c[0],  lw = 2.0, alpha = 0.6 )      # Plots y versus x as a line
    ax1.plot(x2, y2,    label="n=20",   c=c[1],  lw = 2.0, alpha = 0.6  )
    if plot3:
        ax1.plot(x3, y3,    label="ppo2",   c=c[2],  lw = 1.0, alpha = 0.99 )
    ax1.add_patch(Circle(simdata[-1,0:2], 0.55, color='black', alpha=0.9,))
    ax1.add_patch(Circle(target[0:2], 0.2, color='green', alpha=0.9))
    for i in range(ob.shape[0]):
        ax1.add_patch(Circle( ob[i,0:2], ob[i,2], color='red', alpha=0.8)) 
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
    ax2.plot(t,mpct,    label=f"{round(mpcave*1000,1)} ms mean ppo1",    c=c[0],  lw = 1.0 )
    ax2.plot(t2,mpct2,  label=f"{round(mpcave2*1000,1)} ms mean n=20",   c=c[1],   lw = 1.0 )
    if plot3:
        ax2.plot(t3,mpct3,  label=f"{round(mpcave3*1000,1)} ms mean ppo2",   c=c[2], lw = 0.5  )
    ax2.set_ylim(5,400)
    ax2.legend(fontsize=8)

    # ax5.plot(t,n, label="NMPC-N")
    # # ax5.set_ylim(0, 100)
    # ax3b = ax3.twinx()
    # ax3b.plot(t, a, label="action", color='orange')
    # ax3b.set_ylim(0, 1)
    ax3.plot(t,a,   label="ppo1",    c=c[0]         )
    ax3.plot(t2,a2, label="n=20",    c=c[1]         )
    if plot3:
        ax3.plot(t3,a3, label="ppo2",    c=c[2], lw = 0.5)
    # ax3.set_ylim(0,1)
    ax3.legend(fontsize=8)

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
    ax3.set_ylabel('Horizon Length (N)')
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
    # Load datafiles
    with open(f'test_data/simdata-2-1.0-rl-set5.pkl', 'rb') as f: 
        simdata = pickle.load(f)

    with open(f'test_data/simdata-2-1.0-n20.pkl', 'rb') as f: 
        simdata2 = pickle.load(f)

    # with open(f'test_data/simdata-1-4.0-rl-basic.pkl', 'rb') as f: 
    #     simdata3 = pickle.load(f)

    with open(f'test_data/map-2obs-1.0.pkl', 'rb') as f: 
        map = pickle.load(f)

    plotSimdata(simdata,simdata2,simdata3=None,env=map)



