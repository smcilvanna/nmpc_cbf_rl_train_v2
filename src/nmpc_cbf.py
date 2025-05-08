#!/usr/bin/env python3

import numpy as np
import casadi as ca


class NMPC_CBF_MULTI_N:
    def __init__(self, dt, nVals, nObs):
        
        self.dt = dt                                    # time step
        self.nVals = nVals                              # horizon length values for solvers
        self.nrange = len(nVals)
        self.nObs = nObs                                # number of obstacles
        self.wStates = 0.5*np.diag([1.0, 1.0, 0.1])         # Weight matrix for states # Reduced x/y weights from 5.0 → 1.0
        self.wCtrls = np.diag([1.0, 0.5])               # Weight matrix for controls
        self.wTerm =  50*np.diag([5.0, 5.0, 0.1])     # Weight matrix for Terminal state # Reduced from 1e5 to 1e3
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
        self.minCBF = 0.001                              # Set minimum cbf value for normalised actions
        self.maxCBF = 5                               # Set maximum cbf value for normalised actions
    
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
            
            # # NEW: Velocity penalty term
            # target_distance = ca.norm_2(st[:2] - solver["stateTgt"][:2])
            # velocity_error = self.max_v - ct[0]  # Difference from max allowed velocity
            # velocity_penalty = ca.exp(-0.5 * target_distance) * (velocity_error ** 2)

            # objective function across horizon
            stateErr = st - solver["stateTgt"]
            costFunction = costFunction + ca.mtimes([stateErr, self.wStates, stateErr.T]) + ca.mtimes([ct, self.wCtrls, ct.T]) #+ 20.0 * velocity_penalty
        # Terminal cost
        terminal_err = solver["stateHorizon"][-1,:] - solver["stateTgt"]
        costFunction += ca.mtimes([terminal_err, self.wTerm, terminal_err.T])
        solver["opt"].minimize(costFunction)

        # CBF for obstacles (vectorised)
        for i in range(N):
            st = solver["stateHorizon"][i, :]
            st_next = solver["stateHorizon"][i+1, :]
            obs_xy = solver["obstacles"][:, :2]
            cbf = "relaxed" # <<<<<<<<<<<<<<<<<<<<<<<<<<< SET CBF TYPE <<<<<<<<<<<<<<<<<<<<<<<<
            if cbf == "ecbf":
                # >>> ECBF <<< Vectorised
                a = (solver["obstacles"][:, 2] + self.vehRad)**-2
                h = a * ((st[0] - obs_xy[:, 0])**2 + (st[1] - obs_xy[:, 1])**2) - 1
                lfh = 2 * a * ( (st[0] - obs_xy[:, 0]) * (st_next[0] - st[0])/self.dt +
                                (st[1] - obs_xy[:, 1]) * (st_next[1] - st[1])/self.dt   )
                solver["opt"].subject_to(lfh + solver["cbfParms"] * h >= 0)
            elif cbf == "relaxed":
                # >>> Relaxed CBF <<< Vectorised
                h =      ((st[0]      - obs_xy[:, 0])**2 + (st[1]      - obs_xy[:, 1])**2) - (solver["obstacles"][:, 2] + self.vehRad)**2
                h_next = ((st_next[0] - obs_xy[:, 0])**2 + (st_next[1] - obs_xy[:, 1])**2) - (solver["obstacles"][:, 2] + self.vehRad)**2
                solver["opt"].subject_to(h_next - (1 - solver["cbfParms"]) * h >= 0)

        # boundary of state and control input
        solver["opt"].subject_to(solver["opt"].bounded(self.min_x,     solver["stateHorizon"][:,0], self.max_x))
        solver["opt"].subject_to(solver["opt"].bounded(self.min_y,     solver["stateHorizon"][:,1], self.max_y))
        solver["opt"].subject_to(solver["opt"].bounded(self.min_theta, solver["stateHorizon"][:,2], self.max_theta))    
        solver["opt"].subject_to(solver["opt"].bounded(self.min_v,     solver["ctrlHorizon"][:,0], self.max_v))
        solver["opt"].subject_to(solver["opt"].bounded(self.min_omega, solver["ctrlHorizon"][:,1], self.max_omega))

        # setup optimization parameters #max iter was 2000
        opts_setting = {'ipopt.max_iter':10000,
                        'ipopt.print_level':0,
                        'print_time':0,
                        'ipopt.acceptable_tol':1e-4,            # relaxed from 1e-8
                        'ipopt.acceptable_obj_change_tol':1e-5 # relaxed from 1e-6
                        # 'ipopt.warm_start_init_point': 'yes'    # Enable warm starting
                        }
        
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
    
    def setObstacles(self,obstacles):
        for solver in self.solvers:
            solver["opt"].set_value(solver["obstacles"], obstacles )

    def setTarget(self,targetPos):
        for solver in self.solvers:
            solver["opt"].set_value(solver["stateTgt"],  targetPos )

    def solve(self, currentPos, cbfParms):
        self.reset_nmpc(currentPos)     # Clear Warm Start Solutions
        # select the solver
        solver = self.solvers[self.solversIdx]
        # set the parameters
        solver["opt"].set_value(solver["stateNow"],  currentPos)
        solver["opt"].set_value(solver["cbfParms"],  cbfParms  )
        # set the optimisation variables
        solver["opt"].set_initial(solver["stateHorizon"], self.stateHorizon)    # provide the initial guess of state for the next step
        solver["opt"].set_initial(solver["ctrlHorizon"], self.ctrlHorizon)            # provide the initial guess of control for the next step       
        ## solve the problem
        sol = solver["opt"].solve()
        ## obtain the control input
        newCtrlHorizon = sol.value(solver["ctrlHorizon"])
        # self.ctrlHorizon[:-1, :] = newCtrlHorizon[1:, :]
        # self.ctrlHorizon[-1, :] = newCtrlHorizon[-1, :]
        solStateHorizon = sol.value(solver["stateHorizon"])
        self.stateHorizon = np.vstack([solStateHorizon[1:], solStateHorizon[-1:]])
        
        return newCtrlHorizon[0,:]

    def reset_nmpc(self,startPos):           # Reset the NMPC for the next episode
        # # Warm start the state solutions, extending at full velocity towards target
        # dx = self.max_v * np.array([np.cos(target[2]), np.sin(target[2])])
        # steps = np.arange(1, self.currentN+1)[:, None]  # [1,2,...,10] as column vector
        # new_xy = steps * dx                      # Cumulative increments (10,2)
        # new_rows = np.hstack((new_xy, np.full((self.currentN, 1), target[2])))
        # self.stateHorizon = np.vstack((np.array([[0.0, 0.0, target[2]]]), new_rows))

        # # Warm start control horizon, start max v, zero w
        # self.ctrlHorizon = np.column_stack((np.ones(self.currentN), np.zeros(self.currentN)))
        
        self.stateHorizon = np.repeat(startPos.reshape(1,-1), self.currentN+1, axis=0)
        self.ctrlHorizon = np.column_stack((np.ones(self.currentN), np.zeros(self.currentN)))

        return
    
    # def cold_reset_mpc(self,currentPos):
    #     # Warm start the state solutions, extending at full velocity towards target
    #     # dx = 0.00 * np.array([np.cos(target[2]), np.sin(target[2])])
    #     # steps = np.arange(1, self.currentN+1)[:, None]  # [1,2,...,10] as column vector
    #     # new_xy = steps * dx                      # Cumulative increments (10,2)
    #     # new_rows = np.hstack((new_xy, np.full((self.currentN, 1), target[2])))
    #     # self.stateHorizon = np.vstack((np.array([[0.0, 0.0, target[2]]]), new_rows))
    #     # self.ctrlHorizon = np.column_stack((np.ones(self.currentN)*0.1, np.zeros(self.currentN)))
        
    #     # self.stateHorizon = np.zeros((self.currentN+1, 3))
    #     # self.ctrlHorizon = np.zeros((self.currentN,2))
    #     self.stateHorizon = np.repeat(currentPos, self.currentN+1, axis=0)
    #     self.ctrlHorizon = np.zeros((self.currentN,2))
    
    def adjustHorizon(self, action, current_position):
        newIdx = self.indexOfN(action)
        newN = self.nVals[newIdx]
        self.solversIdx = newIdx
        if newN < self.currentN:
            # Shrink horizon
            self.ctrlHorizon = self.ctrlHorizon[0:newN, :]
            self.stateHorizon = self.stateHorizon[0:newN+1, :]
        elif newN > self.currentN:  # Changed from else to explicit check
            self.reset_nmpc(current_position)
        self.currentN = newN

        #     # # Expand horizon
        #     # dx = self.stateHorizon[-1] - self.stateHorizon[-2]
        #     # addState = [self.stateHorizon[-1] + (i+1)*dx 
        #     #         for i in range(newN - self.currentN)]
        #     # addCtrl = [self.ctrlHorizon[-1] 
        #     #         for _ in range(newN - self.currentN)]
            
        #     # # Convert lists to arrays before vstack
        #     # self.ctrlHorizon = np.vstack([self.ctrlHorizon, np.array(addCtrl)])
        #     # self.stateHorizon = np.vstack([self.stateHorizon, np.array(addState)])
        #     self.stateHorizon = np.vstack([self.stateHorizon, np.repeat(self.stateHorizon[-1:], newN - self.currentN, axis=0)])
        #     self.ctrlHorizon = np.vstack([self.ctrlHorizon, np.repeat(self.ctrlHorizon[-1:], newN - self.currentN, axis=0)])
        # No else needed - do nothing when newN == currentN

    # def normalActionsCBF(self, actions):
    #     cbf = actions * (self.maxCBF - self.minCBF) + self.minCBF
    #     return cbf
    
    # def normalActionsN(self, action):
    #     Nindex = round(action * self.nrange)
    #     Nindex = np.clip(Nindex,0,self.nrange-1)
    #     return Nindex


if __name__ =="__main__":
    print("NMPC-CBF Solver Class Definition")
