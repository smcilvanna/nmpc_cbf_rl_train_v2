#!/usr/bin/env python3

import numpy as np
import math
import casadi as ca


class NMPC_CBF_MULTI_N:
    def __init__(self, dt, nRange, nObs):
        
        self.dt = dt                            # time step
        self.nRange = nRange                        # horizon length
        self.nObs = nObs                            # number of obstacles
        self.wStates = np.diag([5.0, 5.0, 0.5])         # Weight matrix for states
        self.wCtrls = np.diag([5.0, 0.1])              # Weight matrix for controls
        self.wTerm = 10**4*np.diag([1.0, 1.0, 0.001]) # Weight matrix for Terminal state
        self.min_x = -100.0
        self.max_x =  100.0
        self.min_y = -100.0
        self.max_y =  100.0
        self.min_theta = -np.inf
        self.max_theta =  np.inf
        self.min_v = -1.0
        self.max_v =  1.0
        self.min_omega = -0.7854
        self.max_omega =  0.7854
        self.max_dv = 1.0
        self.max_domega = 3.14/7
        self.vehRad = 0.55
        self.solvers = []
        self.setup_controllers()
        # self.cbfParms = cbfParms
        # self.obstacles = obstacles
        # self.nObs = len(self.SO[:, 0])
        # Initial value for state and control input
        # self.next_states = np.ones((self.nRange+1, 3))*init_pos
        # self.u0 = np.zeros((self.nRange, 2))


    def setup_controllers(self):
        for N in range(self.nRange):
            self.setup_controller(N)
    
    def setup_controller(self, N):
        
        
        solver = {
            "opt" : ca.Opti()
        }
        solver["stateHorizon"] = self.opt.variable(N+1, 3)
                            #     x = opt_states[:,0]
                            #     y = opt_states[:,1]
                            # theta = opt_states[:,2]
        solver["ctrlHorizon"] = self.opt.variable(N, 2)
                            #     v = opt_controls[:,0]
                            # omega = opt_controls[:,1]
        
        # kinematic model
        solver["f"] = lambda x_, u_: ca.vertcat(*[ca.cos(x_[2])*u_[0], ca.sin(x_[2])*u_[0], u_[1]]),
        
        # these parameters are the reference trajectories of the state and inputs
        # solver["u_ref"] = solver["opt"].parameter(N, 2)
        # solver["x_ref"] = solver["opt"].parameter(self.N+2, 3)
        
        # parameters
        solver["stateNow"]  = solver["opt"].parameter(1, 3)
        solver["stateTgt"]  = solver["opt"].parameter(1, 3)
        solver["obstacles"] = solver["opt"].parameter(self.nObs,3)
        solver["cbfParms"]  = solver["opt"].parameter(self.nObs,1)
        
        # initial state constraint
        solver["opt"].subject_to(solver["stateHorizon"][0,:] == solver["stateNow"])
        
        obj = 0 # initalise objective cost
        
        # N horizon state constraints & cost function
        for i in range(N):            
            st = solver["stateHorizon"][i,:]                # current state
            ct = solver["ctrlHorizon"][i,:]                 # current controls
            k1 = solver["f"](st,ct).T                       # rk4 next state calculation
            k2 = solver["f"](st + self.dt/2*k1, ct).T       #
            k3 = solver["f"](st + self.dt/2*k2, ct).T       #
            k4 = solver["f"](st + self.dt*k3, ct).T         #
            stNext = st + self.dt/6*(k1 + 2*k2 + 2*k3 + k4) # next state
            solver["opt"].subject_to(solver["stateHorizon"][i+1,:] == stNext)

        # cost function
        for i in range(N):
            stateErr = solver["stateHorizon"][i,:] - solver["x_ref"][i+1,:]
            ctrlErr = solver["ctrlHorizon"][i,:] - solver["u_ref"][i,:]
            obj = obj + ca.mtimes([stateErr, self.wStates, stateErr.T]) + ca.mtimes([ctrlErr, self.wCtrls, ctrlErr.T])
        #state_error_N = solver["stateHorizon"][self.N,:] - solver["x_ref"][self.N+1,:]
        #obj = obj + ca.mtimes([state_error_N, self.W_v, state_error_N.T])    
        solver["opt"].minimize(obj)

        # CBF for static obstacles
        for i in range(self.N):
            for j in range(self.nObs):            
                st = solver["stateHorizon"][i,:]
                st_next = solver["stateHorizon"][i+1,:]
                h = (st[0]-self.SO[j,0])**2+(st[1]-self.SO[j,1])**2-(self.vehRad+self.SO[j,2])**2
                h_next = (st_next[0]-self.SO[j,0])**2+(st_next[1]-self.SO[j,1])**2-(self.vehRad+self.SO[j,2])**2
                self.opt.subject_to(h_next-(1-self.cbf_gamma)*h >= 0) 

        # constraint the change of velocity
        for i in range(self.N-1):
            dvel = (solver["ctrlHorizon"][i+1,:] - solver["ctrlHorizon"][i,:])/self.dt
            self.opt.subject_to(self.opt.bounded(-self.max_dv, dvel[0], self.max_dv))
            self.opt.subject_to(self.opt.bounded(-self.max_domega, dvel[1], self.max_domega))

        # boundary of state and control input
        self.opt.subject_to(self.opt.bounded(self.min_x, x, self.max_x))
        self.opt.subject_to(self.opt.bounded(self.min_y, y, self.max_y))
        self.opt.subject_to(self.opt.bounded(self.min_theta, theta, self.max_theta))    
        self.opt.subject_to(self.opt.bounded(self.min_v, v, self.max_v))
        self.opt.subject_to(self.opt.bounded(self.min_omega, omega, self.max_omega))
        
        # setup optimization parameters
        opts_setting = {'ipopt.max_iter':200,'ipopt.print_level':0,'print_time':0,'ipopt.acceptable_tol':1e-8,'ipopt.acceptable_obj_change_tol':1e-6}
        #max iter was 2000
        self.opt.solver('ipopt', opts_setting)
    
    def solve(self, next_trajectories, next_controls):
        
        self.opt.set_value(solver["x_ref"], next_trajectories)       # update feedback state and reference
        self.opt.set_value(solver["u_ref"], next_controls)           # update feedback control and reference
        
        self.opt.set_initial(solver["stateHorizon"], self.next_states) # provide the initial guess of state for the next step
        self.opt.set_initial(solver["ctrlHorizon"], self.u0)        # provide the initial guess of control for the next step       
        ## solve the problem
        sol = self.opt.solve()
        
        # if sol.stats()['return_status'] != 'Solve_Succeeded':
        #     kenny_loggins(f"[NPMC-solver]: ERROR! Solver return status: {sol.stats()['return_status']}")      
        
        ## obtain the control input
        new_u0 = sol.value(solver["ctrlHorizon"])
        self.u0[:-1, :] = new_u0[1:, :]
        self.u0[-1, :] = new_u0[-1, :]
        self.next_states = sol.value(solver["stateHorizon"])
        return new_u0[0,:]

    def reset_nmpc(self, obstacle, cbf_gamma):                              # Reset the NMPC for the next episode
        self.u0 = np.zeros((N, 2))                                          # Reset NMPC internal control variable !! Must do this when resetting the episode or the NMPC will cry
        self.next_states = np.zeros((N+1, 3))                               # Reset NMPC internal state variable  !!  Must do this when resetting the episode or the NMPC will cry
        self.SO = obstacle                                                  # Set the obstacle parameter for NMPC
        self.cbf_gamma = cbf_gamma                                          # Set the CBF parameter for NMPC
        self.nObs = len(obstacle[:, 0])                                     # Set the number of obstacles (should always be 1 for this example)
        self.setup_controller()                                             # Setup the controller optimisation for the next episode
        return
