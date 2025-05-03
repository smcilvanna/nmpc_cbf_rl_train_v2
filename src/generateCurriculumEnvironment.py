import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass
import pickle

@dataclass
class MapLimits:
    x: list
    y: list

# def generate_curriculum_environment(curriculum_level: int, gen_fig: bool = False) -> dict:
#     """Generate curriculum-based environment with precise obstacle gap control."""
    
#     # ===== Curriculum Presets =====
#     presets = {
#         1: {'num_obs': 0, 'grid_size': 10, 'min_gap': 0},
#         2: {'num_obs': 3, 'grid_size': 15, 'min_gap': 1.5},
#         3: {'num_obs': 4, 'grid_size': 20, 'min_gap': 1.5},
#         4: {'num_obs': 8, 'grid_size': 25, 'min_gap': 1.2},
#         5: {'num_obs': 10, 'grid_size': 30, 'min_gap': 1.15}
#     }
#     params = presets[curriculum_level]
    
#     # Initialize parameters
#     clearance = 1.0  # Safety zone radius
#     target_pos = _generate_valid_target(params['grid_size'])
#     obstacles = np.empty((0, 3))
    
#     if curriculum_level >= 2:
#         path_vector = target_pos - np.array([0.0, 0.0])
#         path_length = np.linalg.norm(path_vector)
#         path_dir = path_vector / path_length
#         perpendicular = np.array([-path_dir[1], path_dir[0]])
        
#         def is_valid(pos: np.ndarray, radius: float) -> bool:
#             """Check obstacle clearance from protected zones."""
#             origin_dist = np.linalg.norm(pos)
#             target_dist = np.linalg.norm(pos - target_pos)
#             return (origin_dist > clearance + radius and 
#                     target_dist > clearance + radius)

#         # ===== Place Gate Obstacles =====
#         num_gates = params['num_obs'] // 2
#         gate_positions = np.linspace(0.3, 0.7, num_gates)
        
#         for gate_pos in gate_positions:
#             for _ in range(100):  # Retry loop for valid placement
#                 # Random radii within curriculum constraints
#                 r1 = np.random.uniform(0.1, 2.0)
#                 r2 = np.random.uniform(0.1, 2.0)
                
#                 # Calculate required center spacing for edge-to-edge gap
#                 center_spacing = r1 + r2 + params['min_gap']
#                 base_point = path_dir * (gate_pos * path_length)
#                 offset = perpendicular * (center_spacing / 2)
                
#                 pos1 = base_point + offset
#                 pos2 = base_point - offset
                
#                 if is_valid(pos1, r1) and is_valid(pos2, r2):
#                     obstacles = np.vstack([
#                         obstacles,
#                         [pos1[0], pos1[1], r1],
#                         [pos2[0], pos2[1], r2]
#                     ])
#                     break

#         # ===== Add Extra Obstacles =====
#         remaining = params['num_obs'] - 2 * num_gates
#         for _ in range(remaining):
#             valid = False
#             for _ in range(100):  # Retry attempts
#                 # Random position along path
#                 along_path = np.random.uniform(0.2, 0.8)
#                 lateral = np.random.uniform(-params['min_gap']*2, params['min_gap']*2)
#                 pos = path_dir * (along_path * path_length) + perpendicular * lateral
#                 radius = np.random.uniform(0.1, 2.0)
                
#                 if is_valid(pos, radius):
#                     # Check collisions with existing obstacles
#                     collision = False
#                     for obs in obstacles:
#                         existing_pos = obs[:2]
#                         existing_rad = obs[2]
#                         distance = np.linalg.norm(pos - existing_pos)
#                         if distance < (radius + existing_rad + 0.1):
#                             collision = True
#                             break
                    
#                     if not collision:
#                         obstacles = np.vstack([obstacles, [pos[0], pos[1], radius]])
#                         valid = True
#                         break
#             if not valid:
#                 raise RuntimeError("Failed to place extra obstacle")

#     # Need to pass 6 obstacles, pad any missing with far away obstacles
#     while obstacles.shape[0] < 20:
#         dist = 150                                      # put a dummy obstacle far away
#         angle = np.deg2rad(np.random.randint(0,90))     # randomise angle to obstacle and pos
#         false_obs = np.round(np.array([np.cos(angle)*dist , np.sin(angle)*dist, (np.random.randint(1,101))/10]),1)
#         obstacles = np.vstack([obstacles, false_obs])     # add to stack


#     # ===== Final Environment Setup =====
#     target_yaw = np.arctan2(target_pos[1],target_pos[0])
#     target_pos = np.append(target_pos, target_yaw)
#     out = {
#         'target_pos': target_pos,
#         'obstacles': obstacles,
#         'map_limits': MapLimits(x=[0, params['grid_size']], y=[0, params['grid_size']]),
#         'c_level': curriculum_level,
#         'mpc_req_obs': 6,
#         'obs_in_path': len(obstacles)
#     }
    
#     # Generate visualization
#     if gen_fig:
#         fig, ax = plt.subplots()
#         ax.plot(0, 0, 'bo', markersize=8, label='Start')
#         ax.plot(target_pos[0], target_pos[1], 'gx', markersize=10, label='Target')
        
#         # Draw safety zones
#         ax.add_patch(Circle((0, 0), clearance, color='blue', alpha=0.1))
#         ax.add_patch(Circle(target_pos, clearance, color='green', alpha=0.1))
        
#         # Draw obstacles
#         for obs in obstacles:
#             ax.add_patch(Circle(obs[:2], obs[2], fill=False, color='red'))
        
#         ax.set_xlim(out['map_limits'].x)
#         ax.set_ylim(out['map_limits'].y)
#         ax.set_aspect('equal')
#         ax.legend()
#         out['fig'] = fig
#     else:
#         out['fig'] = None

#     return out

def _generate_valid_target(grid_size: float) -> np.ndarray:
    """Generate target position >25% of grid size from origin."""
    min_coord = grid_size * 0.25
    ridx = np.random.randint(0,2)
    target = np.zeros(2)
    target[ridx] = np.random.uniform(min_coord, grid_size)
    target[1-ridx] = grid_size
    return target


#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

def setGate(start_pos,dist,rad,gap,offset,ngates):
    r = [rad , rad]
    gate=np.empty((0,3))
    for i in range(ngates):
        gatei, gtgt = place_obstacles(start_pos, radii=r, dist_to_gap= dist, offset_perp=-offset, gap_size=(gap + 4*i*rad + 0.8*i) )
        gate = np.vstack([gate,gatei])    
    return gate, gtgt

def genCurEnv_2(curriculum_level, gen_fig=False, maxObs=5):
    # Set Target and Start Positions

    if curriculum_level > 2:
        print("Levels above 2 not implemented yet")
        exit()

    if curriculum_level == 1:
        # two mostly overlapping obstacles along path
        grid = 30
        gateDist = 15
        gateOffset = 0.0
        nGates = 1
        obsRad = np.round(np.random.uniform(0.1,10.1),1)
        gateGap = -obsRad*np.random.uniform(1.9, 2.001)
        while abs(gateOffset)  < 0.05:
            gateOffset = np.random.uniform(-(obsRad*0.7),(obsRad*0.7)) 
        
    elif curriculum_level == 2:
        # two obstacles with gap
        grid = 30
        gateDist = np.round(np.random.uniform(15,25))
        nGates = 1
        obsRad = np.round(np.random.uniform(0.1,10.1),1)
        gateGap = np.random.uniform(2.0, 8.0)
        gateOffset = 0.0
        while abs(gateOffset)  < 0.1:
            gateOffset = np.round(np.random.uniform(-(obsRad*0.99),(obsRad*0.99)),2)

    elif curriculum_level == 3:  # NOT IMPLEMENTED
        grid = 40
        gateDist = np.round(np.random.uniform(15,25))
        nGates = 5
        obsRad = np.round(np.random.uniform(0.1,10.1),1)
        gateGap = np.random.uniform(1.2, 2.5)
        gateOffset = 0.0
        while abs(gateOffset)  < obsRad*0.4:
            gateOffset = np.round(np.random.uniform(-(obsRad*3.5),(obsRad*3.5)),1)
    elif curriculum_level == 4:
        grid = 50
    else:
        print("[ERROR] Invalid curriculum level.")
        exit()

    target_pos = _generate_valid_target(grid)
    target_yaw = np.arctan2(target_pos[1],target_pos[0])
    target_pos = np.append(target_pos, target_yaw)
    start_pos = np.array([0.0, 0.0, target_yaw])
    

    if curriculum_level in [1,2]:
        obstacles, gtgt = setGate(start_pos,dist=gateDist,rad=obsRad,gap=gateGap,offset=gateOffset, ngates=nGates)
        passTarget = gtgt
        passTarget[0] += np.sign(gateOffset)*((2*obsRad)+0.55+ 0.5*gateGap)*np.cos(target_yaw+np.pi/2) 
        passTarget[1] += np.sign(gateOffset)*((2*obsRad)+0.55+ 0.5*gateGap)*np.sin(target_yaw+np.pi/2)
        passTarget = [passTarget.tolist()]
    
    elif curriculum_level == 3:
        obstacles, gtgt = setGate(start_pos,dist=gateDist,rad=obsRad,gap=gateGap,offset=gateOffset, ngates=nGates)
        passTarget = [gtgt.tolist()]
    
    elif curriculum_level == 4:
        gateDist = np.round(np.random.uniform(12,22))
        nGates = 5
        obsRad = np.round(np.random.uniform(0.5,10.1),1)
        gateGap = np.random.uniform(1.2, 2.5)
        gateOffset = 0.0
        while abs(gateOffset)  < obsRad*0.4:
            gateOffset = np.round(np.random.uniform(-(obsRad*3.5),(obsRad*3.5)),1)
        obstacles1, gtgt1 = setGate(start_pos,dist=gateDist,rad=obsRad,gap=gateGap,offset=gateOffset, ngates=nGates)

        # second gap
        passDist = np.linalg.norm(target_pos[0:2]-gtgt1)
        print(passDist)
        gateDist = np.round(np.random.uniform(passDist*0.25,passDist*0.75))
        nGates = 5
        obsRad = np.round(np.random.uniform(0.5,(10.5-obsRad)),1)
        gateGap = np.random.uniform(1.2, 2.5)
        gate1offset = gateOffset
        gateOffset = 0.0
        while abs(gateOffset)  < obsRad*0.4 and np.sign(gateOffset) != np.sign(gate1offset):
            gateOffset = np.round(np.random.uniform(-(obsRad*3.5),(obsRad*3.5)),1)
        gapStart = np.array([gtgt1[0], gtgt1[1], np.arctan2( target_pos[1]-gtgt1[1], target_pos[0]-gtgt1[0])])
        obstacles2, gtgt2 = setGate(gapStart,dist=gateDist,rad=obsRad,gap=gateGap,offset=gateOffset, ngates=nGates)
        obstacles = np.vstack([obstacles1, obstacles2])
        passTarget = [gtgt1.tolist(),gtgt2.tolist()]


    while obstacles.shape[0] < maxObs:
        dist = 150                                      # put a dummy obstacle far away
        angle = np.deg2rad(np.random.randint(0,90))     # randomise angle to obstacle and pos
        false_obs = np.round(np.array([np.cos(angle)*dist , np.sin(angle)*dist, (np.random.randint(1,101))/10]),1)
        obstacles = np.vstack([obstacles, false_obs])     # add to stack
    
    startDist = np.linalg.norm(target_pos)

    out = {
        'target_pos': target_pos,
        'obstacles': obstacles,
        'pass_targets' : passTarget,
        'startDist' : startDist
        }


    if gen_fig:
        fig, ax = plt.subplots()
        # ax.plot(0, 0, 'bo', markersize=8, label='Start')
        ax.plot(target_pos[0], target_pos[1], 'gx', markersize=10, label='Target')
        
        # Draw safety zones
        ax.add_patch(Circle((0, 0), 1.0, color='blue', alpha=0.1))      # vehicle start clearance
        ax.add_patch(Circle((0, 0), 0.55, color='black', alpha=0.8))      # vehicle start clearance
        ax.add_patch(Circle(target_pos, 0.5, color='green', alpha=0.1)) # finish clearance
        
        for pgt in passTarget:
            ax.add_patch(Circle(pgt, 0.55, color='green',alpha=0.9))
        
        # Draw obstacles
        for obs in obstacles:
            ax.add_patch(Circle(obs[:2], obs[2], fill=False, color='red'))
        
        ax.set_xlim((-10,grid))
        ax.set_ylim((-10,grid))
        ax.set_aspect('equal')
        # ax.legend()
        out["fig"] = fig
        plt.show()
    else:
        out["fig"] = None

    
    

    return out


def place_obstacles(start_pose, radii, dist_to_gap, offset_perp, gap_size):
    """
    Places two circular obstacles forming a gate relative to a starting pose.
    
    Parameters:
    start_pose (list): [x, y, theta] initial position and orientation (radians)
    radii (list): [radius1, radius2] sizes of the two obstacles
    dist_to_gap (float): Distance from start pose to gap center along theta
    offset_perp (float): Perpendicular offset from theta axis to gap center
    gap_size (float): Clear space between obstacle edges
    
    Returns:
    np.array: 2x3 array of [x, y, radius] for each obstacle
    """
    x0, y0, theta = start_pose
    r1, r2 = radii
    
    # Calculate gap center position
    gap_center = np.array([
        x0 + dist_to_gap * np.cos(theta),
        y0 + dist_to_gap * np.sin(theta)
    ])
    
    # Calculate perpendicular direction vector
    perp_dir = np.array([-np.sin(theta), np.cos(theta)])
    
    # Apply perpendicular offset
    gap_center += offset_perp * perp_dir
    
    # Calculate obstacle separation (center-to-center distance)
    center_to_center = r1 + r2 + gap_size
    
    # Place obstacles symmetrically about gap center
    obs1_pos = gap_center - (center_to_center/2) * perp_dir
    obs2_pos = gap_center + (center_to_center/2) * perp_dir
    
    obstacles = np.array([ [obs1_pos[0], obs1_pos[1], r1],
                           [obs2_pos[0], obs2_pos[1], r2] ])
    
    return obstacles, gap_center


if __name__ == "__main__":
    # Test all curriculum levels
    # for level in range(5, 6):
    #     try:
    #         env = generate_curriculum_environment(level, gen_fig=True)
    #         plt.title(f"Curriculum Level {level}")
    #         plt.show()
    #         print(f"Level {level} obstacles:\n{env['obstacles']}")
    #     except Exception as e:
    #         print(f"Error generating level {level}: {str(e)}")

    # env = generate_curriculum_environment(2, gen_fig=True)
    # plt.show()

    env = genCurEnv_2(curriculum_level=1 , gen_fig=True)

    input("ENTER to save file")
    # Save to file
    with open('env1-1.pkl', 'wb') as f:
        pickle.dump(env, f)


