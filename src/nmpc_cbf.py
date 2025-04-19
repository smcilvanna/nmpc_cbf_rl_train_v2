#!/usr/bin/env python3

import numpy as np
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
            "opt" : ca.Opti(),
            "N"   : N                                               #     x = stateHorizon[:,0]
        }                                                           #     y = stateHorizon[:,1]
        solver["stateHorizon"] = solver["opt"].variable(N+1, 3)     # theta = stateHorizon[:,2]
        solver["ctrlHorizon"] = solver["opt"].variable(N, 2)        #     v = ctrlHorizon[:,0]
                                                                    # omega = ctrlHorizon[:,1]

        # kinematic model definition f
        solver["f"] = lambda x_, u_: ca.vertcat(*[ca.cos(x_[2])*u_[0], ca.sin(x_[2])*u_[0], u_[1]])
        
        # these parameters are the reference trajectories of the state and inputs (removed for point tracking only)
        # solver["u_ref"] = solver["opt"].parameter(N, 2)
        # solver["x_ref"] = solver["opt"].parameter(self.N+2, 3)
        
        # parameters
        solver["stateNow"]  = solver["opt"].parameter(1, 3)
        solver["stateTgt"]  = solver["opt"].parameter(1, 3)
        solver["obstacles"] = solver["opt"].parameter(self.nObs,3)
        solver["cbfParms"]  = solver["opt"].parameter(self.nObs,1)
        
        # initial state constraint
        solver["opt"].subject_to(solver["stateHorizon"][0,:] == solver["stateNow"])
        
        costFunction = 0 # initalise objective cost
        
        for i in range(N):            
            # N horizon state constraints
            st = solver["stateHorizon"][i,:]                # current state
            ct = solver["ctrlHorizon"][i,:]                 # current controls
            k1 = solver["f"](st,ct).T                       # rk4 next state calculation
            k2 = solver["f"](st + self.dt/2*k1, ct).T       #
            k3 = solver["f"](st + self.dt/2*k2, ct).T       #
            k4 = solver["f"](st + self.dt*k3, ct).T         #
            stNext = st + self.dt/6*(k1 + 2*k2 + 2*k3 + k4) # next state
            solver["opt"].subject_to(solver["stateHorizon"][i+1,:] == stNext)
            
            # objective function across horizon
            stateErr = st - solver["stateTgt"]
            costFunction = costFunction + ca.mtimes([stateErr, self.wStates, stateErr.T]) + ca.mtimes([ct, self.wCtrls, ct.T])
        solver["opt"].minimize(costFunction)

        # Relaxed CBF for obstacles
        for i in range(N):
            for j in range(self.nObs):            
                st = solver["stateHorizon"][i,0:2]          # current state xy position
                st_next = solver["stateHorizon"][i+1,0:2]   # next state xy position

                # h      = (     st[0]-solver["obstacles"][j,0])**2 + (st[1]-solver["obstacles"][j,1])**2-(self.vehRad+solver["obstacles"][j,2])**2
                # h_next = (st_next[0]-solver["obstacles"][j,0])**2 + (st_next[1]-self.SO[j,1])**2-(self.vehRad+self.SO[j,2])**2
                
                # relaxed cbf components
                h      = ca.norm_2(st      - solver["obstacles"][j,0:2]) - (self.vehRad + solver["obstacles"][j,2])
                h_next = ca.norm_2(st_next - solver["obstacles"][j,0:2]) - (self.vehRad + solver["obstacles"][j,2])
                # relaxed cbf constraint for obstacle j at horizon step i
                solver["opt"].subject_to( h_next - (1- solver["cbfParms"][j] )*h >= 0) 

        # # constraint the change of velocity (removed)
        # for i in range(self.N-1):
        #     dvel = (solver["ctrlHorizon"][i+1,:] - solver["ctrlHorizon"][i,:])/self.dt
        #     self.opt.subject_to(self.opt.bounded(-self.max_dv, dvel[0], self.max_dv))
        #     self.opt.subject_to(self.opt.bounded(-self.max_domega, dvel[1], self.max_domega))

        # boundary of state and control input
        solver["opt"].subject_to(solver["opt"].bounded(self.min_x,     solver["stateHorizon"][:,0], self.max_x))
        solver["opt"].subject_to(solver["opt"].bounded(self.min_y,     solver["stateHorizon"][:,1], self.max_y))
        solver["opt"].subject_to(solver["opt"].bounded(self.min_theta, solver["stateHorizon"][:,2], self.max_theta))    
        solver["opt"].subject_to(solver["opt"].bounded(self.min_v,     solver["ctrlHorizon"][:,0], self.max_v))
        solver["opt"].subject_to(solver["opt"].bounded(self.min_omega, solver["ctrlHorizon"][:,1], self.max_omega))
        
        # setup optimization parameters #max iter was 2000
        opts_setting = {'ipopt.max_iter':200,
                        'ipopt.print_level':0,
                        'print_time':0,
                        'ipopt.acceptable_tol':1e-8,
                        'ipopt.acceptable_obj_change_tol':1e-6}
        
        solver["opt"].solver('ipopt', opts_setting)
        self.solvers.append(solver)     # add this solver to the solver stack
    
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


if __name__ == "__main__":
    nmpc = NMPC_CBF_MULTI_N(0.1, 10, 3)
    print("NMPC_CBF_MULTI_N class initialized successfully.")
    # # Example usage
    # state_now = np.array([0.0, 0.0, 0.0])
    # state_tgt = np.array([1.0, 1.0, 0.0])
    # obstacles = np.array([[2.0, 2.0, 0.5]])
    # cbf_parms = np.array([[1.0]])
    
    # nmpc.setup_controller(10)
    # nmpc.solve(state_now, state_tgt, obstacles, cbf_parms)