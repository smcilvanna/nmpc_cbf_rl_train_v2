#!/usr/bin/env python3

import numpy as np
import math
import casadi as ca


class NMPC_CBF_MULTI_N:
    def __init__(self, limits, dt, nRange, wState, wCtrl, wTerm, cbfParms, obstacles, init_pos):
        
        
        
        self.dt = dt                            # time step
        self.nRange = nRange                        # horizon length
        self.W_q = wState                          # Weight matrix for states
        self.W_r = wCtrl                          # Weight matrix for controls
        self.W_v = wTerm                          # Weight matrix for Terminal state
        self.min_x = limits[0] 
        self.max_x = limits[1]
        self.min_y = limits[2] 
        self.max_y = limits[3]
        self.min_theta = limits[4]
        self.max_theta = limits[5]
        self.min_v = limits[6]
        self.max_v = limits[7]
        self.min_omega = limits[8]
        self.max_omega = limits[9]
        self.max_dv = 1.0
        self.max_domega = 3.14/7
        self.R_husky = 0.55             ########################################################################
        self.cbfParms = cbfParms
        self.obstacles = obstacles
        self.nObs = len(self.SO[:, 0])
        # Initial value for state and control input
        self.next_states = np.ones((self.nRange+1, 3))*init_pos
        self.u0 = np.zeros((self.nRange, 2))
        self.setup_controllers()


    def setup_controllers(self):
    
    def setup_controller(self):
        self.opt = ca.Opti()
        self.opt_states = self.opt.variable(self.nRange+1, 3)
        x = self.opt_states[:,0]
        y = self.opt_states[:,1]
        theta = self.opt_states[:,2]
        self.opt_controls = self.opt.variable(self.nRange, 2)
        v = self.opt_controls[:,0]
        omega = self.opt_controls[:,1]

        # dynamic mapping function
        f = lambda x_, u_: ca.vertcat(*[ca.cos(x_[2])*u_[0],  
                                        ca.sin(x_[2])*u_[0], 
                                        u_[1]])      
                                              
        # these parameters are the reference trajectories of the state and inputs
        self.u_ref = self.opt.parameter(self.nRange, 2)
        self.x_ref = self.opt.parameter(self.nRange+2, 3)

        # dynamic constraint
        self.opt.subject_to(self.opt_states[0,:] == self.x_ref[0,:])
        for i in range(self.nRange):            
            st = self.opt_states[i,:]
            ct = self.opt_controls[i,:]
            k1 = f(st,ct).T
            k2 = f(st + self.dt/2*k1, ct).T
            k3 = f(st + self.dt/2*k2, ct).T
            k4 = f(st + self.dt*k3, ct).T
            x_next = st + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)
            self.opt.subject_to(self.opt_states[i+1,:] == x_next)

        # cost function
        obj = 0
        for i in range(self.nRange):
            state_error_ = self.opt_states[i,:] - self.x_ref[i+1,:]
            control_error_ = self.opt_controls[i,:] - self.u_ref[i,:]
            obj = obj + ca.mtimes([state_error_, self.W_q, state_error_.T]) + ca.mtimes([control_error_, self.W_r, control_error_.T])
        #state_error_N = self.opt_states[self.N,:] - self.x_ref[self.N+1,:]
        #obj = obj + ca.mtimes([state_error_N, self.W_v, state_error_N.T])    
        self.opt.minimize(obj)

        # CBF for static obstacles
        for i in range(self.nRange):
            for j in range(self.nObs):            
                st = self.opt_states[i,:]
                st_next = self.opt_states[i+1,:]
                h = (st[0]-self.SO[j,0])**2+(st[1]-self.SO[j,1])**2-(self.R_husky+self.SO[j,2])**2
                h_next = (st_next[0]-self.SO[j,0])**2+(st_next[1]-self.SO[j,1])**2-(self.R_husky+self.SO[j,2])**2
                self.opt.subject_to(h_next-(1-self.cbf_gamma)*h >= 0) 

        # constraint the change of velocity
        for i in range(self.nRange-1):
            dvel = (self.opt_controls[i+1,:] - self.opt_controls[i,:])/self.dt
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
        
        self.opt.set_value(self.x_ref, next_trajectories)       # update feedback state and reference
        self.opt.set_value(self.u_ref, next_controls)           # update feedback control and reference
        
        self.opt.set_initial(self.opt_states, self.next_states) # provide the initial guess of state for the next step
        self.opt.set_initial(self.opt_controls, self.u0)        # provide the initial guess of control for the next step       
        ## solve the problem
        sol = self.opt.solve()
        
        # if sol.stats()['return_status'] != 'Solve_Succeeded':
        #     kenny_loggins(f"[NPMC-solver]: ERROR! Solver return status: {sol.stats()['return_status']}")      
        
        ## obtain the control input
        new_u0 = sol.value(self.opt_controls)
        self.u0[:-1, :] = new_u0[1:, :]
        self.u0[-1, :] = new_u0[-1, :]
        self.next_states = sol.value(self.opt_states)
        return new_u0[0,:]

    def reset_nmpc(self, obstacle, cbf_gamma):                              # Reset the NMPC for the next episode
        self.u0 = np.zeros((N, 2))                                          # Reset NMPC internal control variable !! Must do this when resetting the episode or the NMPC will cry
        self.next_states = np.zeros((N+1, 3))                               # Reset NMPC internal state variable  !!  Must do this when resetting the episode or the NMPC will cry
        self.SO = obstacle                                                  # Set the obstacle parameter for NMPC
        self.cbf_gamma = cbf_gamma                                          # Set the CBF parameter for NMPC
        self.nObs = len(obstacle[:, 0])                                     # Set the number of obstacles (should always be 1 for this example)
        self.setup_controller()                                             # Setup the controller optimisation for the next episode
        return
