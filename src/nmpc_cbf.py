#!/usr/bin/env python3

import numpy as np
import casadi as ca


class NMPC_CBF_MULTI_N:
    def __init__(self, dt, nVals, nObs):
        
        self.dt = dt                                    # time step
        self.nVals = nVals                              # horizon length values for solvers
        self.nObs = nObs                                # number of obstacles
        self.wStates = np.diag([5.0, 5.0, 0.1])         # Weight matrix for states
        self.wCtrls = np.diag([5.0, 1.0])               # Weight matrix for controls
        self.wTerm = 1e5*np.diag([1.0, 1.0, 0.001])   # Weight matrix for Terminal state
        self.min_x = -100.0                             # State bounds
        self.max_x =  100.0                             
        self.min_y = -100.0                             
        self.max_y =  100.0                             
        self.min_theta = -np.inf                        
        self.max_theta =  np.inf                        
        self.min_v = -1.0                               # Linear velocity control bounds
        self.max_v =  1.0                               
        self.min_omega = -0.7854                        # Angular velocity control bounds
        self.max_omega =  0.7854                        
        self.max_dv = 1.0                               # Acceleration bounds (not used)
        self.max_domega = 3.14/7                        
        self.vehRad = 0.55                              # Vehicle clearance radius for obstacle collision detection
        self.solvers = []                               # Init empty solver stack for variable N solvers
        self.setup_controllers()                        # Create solvers from nRange list input arg
        self.solversIdx = 0                             # Set the index of the active solver
        self.currentN = nVals[0]                        # Set the N horizon length of active solver
        self.stateHorizon = np.array([])                # Init empty array for state horizon
        self.ctrlHorizon = np.array([])                 # Init empty array for control horizon
    
    def setup_controllers(self):
        for N in self.nVals:
            self.setup_controller(N)
    
    def setup_controller(self, N):
        
        solver = {
            "opt" : ca.Opti(),
            "N"   : N
        }
        solver["stateHorizon"] = solver["opt"].variable(N+1, 3)     # x:[:,0] | y:[:,1] | theta:[:,3]
        solver["ctrlHorizon"]  = solver["opt"].variable(N, 2)        # v:[:,0] | omega:[:,1]

        # kinematic model definition f
        solver["f"] = lambda x_, u_: ca.vertcat(*[ca.cos(x_[2])*u_[0], ca.sin(x_[2])*u_[0], u_[1]])
        # # Define the ODE function
        # x = ca.MX.sym('x', 3)
        # u = ca.MX.sym('u', 2)
        # ode = solver["f"](x, u)
        # dae = {'x': x, 'p': u, 'ode': ode}  #       t0   dt         opts
        # integrator = ca.integrator('F', 'cvodes', dae, 0, self.dt)

        # parameters to be passsed at solve time
        solver["stateNow"]  = solver["opt"].parameter(1, 3)
        solver["stateTgt"]  = solver["opt"].parameter(1, 3)
        solver["obstacles"] = solver["opt"].parameter(self.nObs,3)
        solver["cbfParms"]  = solver["opt"].parameter(self.nObs,1)
        
        # initial state constraint
        solver["opt"].subject_to(solver["stateHorizon"][0,:] == solver["stateNow"])
        
        costFunction = 0 # initalise objective cost
        
        for i in range(N):            
            # N horizon state constraints
            st = solver["stateHorizon"][i,:]                    # current state
            ct = solver["ctrlHorizon"][i,:]                     # current controls
            k1 = solver["f"](st,ct).T                       # rk4 next state calculation
            k2 = solver["f"](st + self.dt/2*k1, ct).T       #
            k3 = solver["f"](st + self.dt/2*k2, ct).T       #
            k4 = solver["f"](st + self.dt*k3, ct).T         #
            stNext = st + self.dt/6*(k1 + 2*k2 + 2*k3 + k4) # next state
            # stNext = integrator(x0=st, p=ct)["xf"]  # Next state i+1
            solver["opt"].subject_to(solver["stateHorizon"][i+1, :] == stNext)  # append state constraint for horizon step i+1
            
            # objective function across horizon
            stateErr = st - solver["stateTgt"]
            costFunction = costFunction + ca.mtimes([stateErr, self.wStates, stateErr.T]) + ca.mtimes([ct, self.wCtrls, ct.T])
        solver["opt"].minimize(costFunction)

        # Relaxed CBF for obstacles
        for i in range(N):
            st      = ca.repmat(solver["stateHorizon"][i  ,0:2],self.nObs,1)    # current state xy position
            st_next = ca.repmat(solver["stateHorizon"][i+1,0:2],self.nObs,1)    # next state xy position
            voRads = solver["obstacles"][:, 2] + self.vehRad
            h      = ca.sqrt( ca.sum2((     st - solver["obstacles"][:, 0:2])**2) ) - voRads
            h_next = ca.sqrt( ca.sum2((st_next - solver["obstacles"][:, 0:2])**2) ) - voRads
            # Apply constraints for all obstacles at horizon step i
            solver["opt"].subject_to(h_next - (1 - solver["cbfParms"]) * h >= 0)


        # boundary of state and control input
        solver["opt"].subject_to(solver["opt"].bounded(self.min_x,     solver["stateHorizon"][:,0], self.max_x))
        solver["opt"].subject_to(solver["opt"].bounded(self.min_y,     solver["stateHorizon"][:,1], self.max_y))
        solver["opt"].subject_to(solver["opt"].bounded(self.min_theta, solver["stateHorizon"][:,2], self.max_theta))    
        solver["opt"].subject_to(solver["opt"].bounded(self.min_v,     solver["ctrlHorizon"][:,0], self.max_v))
        solver["opt"].subject_to(solver["opt"].bounded(self.min_omega, solver["ctrlHorizon"][:,1], self.max_omega))
        
        # setup optimization parameters #max iter was 2000
        opts_setting = {'ipopt.max_iter':500,
                        'ipopt.print_level':0,
                        'print_time':0,
                        'ipopt.acceptable_tol':1e-8,
                        'ipopt.acceptable_obj_change_tol':1e-6}
        
        solver["opt"].solver('ipopt', opts_setting)
        self.solvers.append(solver)     # append this N horizon solver to the solver stack
    
    def indexOfN(self,N):
        # find the solvers index of a given N value
        # index = next(i for i, s in enumerate(nmpc.solvers) if s.get("N") == N)        
        try:
            index = self.nVals.index(N)
        except:
            print(f"Couldn't Find {N} Horizon Length Solver. END")
            exit()
        return index

    def solve(self, targetPos, currentPos, obstacles, cbfParms):
        # On first step init state and control horizon arrays
        if self.ctrlHorizon.size == 0 and self.stateHorizon.size == 0:
            self.stateHorizon = np.zeros((self.currentN+1, 3))
            self.ctrlHorizon  = np.zeros((self.currentN,   2))
        # select the solver
        solver = self.solvers[self.solversIdx]
        # set the parameters
        solver["opt"].set_value(solver["stateNow"],  currentPos)
        solver["opt"].set_value(solver["stateTgt"],  targetPos )
        solver["opt"].set_value(solver["obstacles"], obstacles )
        solver["opt"].set_value(solver["cbfParms"],  cbfParms  )
        # set the optimisation variables
        solver["opt"].set_initial(solver["stateHorizon"], self.stateHorizon)    # provide the initial guess of state for the next step
        solver["opt"].set_initial(solver["ctrlHorizon"], self.ctrlHorizon)            # provide the initial guess of control for the next step       
        ## solve the problem
        sol = solver["opt"].solve()
                
        ## obtain the control input
        newCtrlHorizon = sol.value(solver["ctrlHorizon"])
        self.ctrlHorizon[:-1, :] = newCtrlHorizon[1:, :]
        self.ctrlHorizon[-1, :] = newCtrlHorizon[-1, :]
        solStateHorizon = sol.value(solver["stateHorizon"])
        self.stateHorizon = np.vstack([solStateHorizon[1:], solStateHorizon[-1:]])
        return newCtrlHorizon[0,:]

    def reset_nmpc(self):           # Reset the NMPC for the next episode
        self.ctrlHorizon = []           # empty horizon arrays
        self.stateHorizon = []
        return
    
    # def adjustHorizon(self,newN):
    #     if newN <= self.currentN:
    #         self.ctrlHorizon  = self.ctrlHorizon[]
    #         self.stateHorizon = 
    #     else:


if __name__ == "__main__":
    nmpc = NMPC_CBF_MULTI_N(0.1, [10, 20, 30, 40, 50], 3)
    print("NMPC_CBF_MULTI_N class initialized successfully.")
    nmpc.solversIdx = 0
    nmpc.currentN = nmpc.nVals[nmpc.solversIdx]
    obstacles = np.array([[20,19.9,1],[220,20,1],[330,30,1]])
    targetPos = np.array([50,50,0.7])
    currentPos = np.array([0,0,0.7])
    cbf = np.array([ 0.999, 0.1 ,0.1])

    # obstacles = obstacles[-1,:]
    # cbf = cbf[-1]

    import time
    from matplotlib import pyplot as plt

    simRealTime = 30
    simSteps = int(simRealTime / 0.1)
    simdata = np.zeros(( simSteps + 1, 5))
    # input(f"Simulate {simRealTime} seconds, {simSteps} steps")
    for i in range(simSteps):
        t = time.time()
        simdata[i,2:] = currentPos
        u = nmpc.solve(targetPos,currentPos,obstacles,cbf)
        currentPos = nmpc.stateHorizon[1,:]
        simdata[i,0] = i
        simdata[i,1] = time.time() - t
        print(f"{i} : {u}")
    
    # print(simdata)
    fig, ax = plt.subplots()
    x = simdata[:,2]
    y = simdata[:,3]
    ax.plot(x, y)      # Plots y versus x as a line

    circle = plt.Circle((obstacles[0,0:2]), radius=1, color='red', fill=False)  # fill=False for outline only
    ax.add_artist(circle)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y vs x')
    plt.show()

    # u = nmpc.solve(targetPos,currentPos,obstacles,cbf)
    print(u)
    print(nmpc.stateHorizon[-1,:])