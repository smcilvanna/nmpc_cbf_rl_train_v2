import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass

@dataclass
class MapLimits:
    x: list
    y: list

def generate_curriculum_environment(curriculum_level: int, gen_fig: bool = False) -> dict:
    """Generate curriculum-based environment with precise obstacle gap control."""
    # ===== Curriculum Presets =====
    presets = {
        1: {'num_obs': 0, 'grid_size': 10, 'min_gap': 0},
        2: {'num_obs': 3, 'grid_size': 15, 'min_gap': 1.5},
        3: {'num_obs': 4, 'grid_size': 20, 'min_gap': 1.5},
        4: {'num_obs': 8, 'grid_size': 25, 'min_gap': 1.2},
        5: {'num_obs': 10, 'grid_size': 30, 'min_gap': 1.15}
    }
    params = presets[curriculum_level]
    
    # Initialize parameters
    veh_rad = 0.5
    clearance = 1.0  # Safety zone radius
    target_pos = _generate_valid_target(params['grid_size'])
    obstacles = np.empty((0, 3))
    
    if curriculum_level >= 2:
        path_vector = target_pos - np.array([0.0, 0.0])
        path_length = np.linalg.norm(path_vector)
        path_dir = path_vector / path_length
        perpendicular = np.array([-path_dir[1], path_dir[0]])
        
        def is_valid(pos: np.ndarray, radius: float) -> bool:
            """Check obstacle clearance from protected zones."""
            origin_dist = np.linalg.norm(pos)
            target_dist = np.linalg.norm(pos - target_pos)
            return (origin_dist > clearance + radius and 
                    target_dist > clearance + radius)

        # ===== Place Gate Obstacles =====
        num_gates = params['num_obs'] // 2
        gate_positions = np.linspace(0.3, 0.7, num_gates)
        
        for gate_pos in gate_positions:
            for _ in range(100):  # Retry loop for valid placement
                # Random radii within curriculum constraints
                r1 = np.random.uniform(0.1, 2.0)
                r2 = np.random.uniform(0.1, 2.0)
                
                # Calculate required center spacing for edge-to-edge gap
                center_spacing = r1 + r2 + params['min_gap']
                base_point = path_dir * (gate_pos * path_length)
                offset = perpendicular * (center_spacing / 2)
                
                pos1 = base_point + offset
                pos2 = base_point - offset
                
                if is_valid(pos1, r1) and is_valid(pos2, r2):
                    obstacles = np.vstack([
                        obstacles,
                        [pos1[0], pos1[1], r1],
                        [pos2[0], pos2[1], r2]
                    ])
                    break

        # ===== Add Extra Obstacles =====
        remaining = params['num_obs'] - 2 * num_gates
        for _ in range(remaining):
            valid = False
            for _ in range(100):  # Retry attempts
                # Random position along path
                along_path = np.random.uniform(0.2, 0.8)
                lateral = np.random.uniform(-params['min_gap']*2, params['min_gap']*2)
                pos = path_dir * (along_path * path_length) + perpendicular * lateral
                radius = np.random.uniform(0.1, 2.0)
                
                if is_valid(pos, radius):
                    # Check collisions with existing obstacles
                    collision = False
                    for obs in obstacles:
                        existing_pos = obs[:2]
                        existing_rad = obs[2]
                        distance = np.linalg.norm(pos - existing_pos)
                        if distance < (radius + existing_rad + 0.1):
                            collision = True
                            break
                    
                    if not collision:
                        obstacles = np.vstack([obstacles, [pos[0], pos[1], radius]])
                        valid = True
                        break
            if not valid:
                raise RuntimeError("Failed to place extra obstacle")

    # Need to pass 6 obstacles, pad any missing with far away obstacles
    while obstacles.shape[0] < 20:
        dist = 150                                      # put a dummy obstacle far away
        angle = np.deg2rad(np.random.randint(0,90))     # randomise angle to obstacle and pos
        false_obs = np.round(np.array([np.cos(angle)*dist , np.sin(angle)*dist, (np.random.randint(1,101))/10]),1)
        obstacles = np.vstack([obstacles, false_obs])     # add to stack


    # ===== Final Environment Setup =====
    target_yaw = np.arctan2(target_pos[1],target_pos[0])
    target_pos = np.append(target_pos, target_yaw)
    out = {
        'target_pos': target_pos,
        'obstacles': obstacles,
        'map_limits': MapLimits(x=[0, params['grid_size']], y=[0, params['grid_size']]),
        'c_level': curriculum_level,
        'mpc_req_obs': 6,
        'obs_in_path': len(obstacles)
    }
    
    # Generate visualization
    if gen_fig:
        fig, ax = plt.subplots()
        ax.plot(0, 0, 'bo', markersize=8, label='Start')
        ax.plot(target_pos[0], target_pos[1], 'gx', markersize=10, label='Target')
        
        # Draw safety zones
        ax.add_patch(Circle((0, 0), clearance, color='blue', alpha=0.1))
        ax.add_patch(Circle(target_pos, clearance, color='green', alpha=0.1))
        
        # Draw obstacles
        for obs in obstacles:
            ax.add_patch(Circle(obs[:2], obs[2], fill=False, color='red'))
        
        ax.set_xlim(out['map_limits'].x)
        ax.set_ylim(out['map_limits'].y)
        ax.set_aspect('equal')
        ax.legend()
        out['fig'] = fig
    else:
        out['fig'] = None

    return out

def _generate_valid_target(grid_size: float) -> np.ndarray:
    """Generate target position >25% of grid size from origin."""
    min_coord = grid_size * 0.25
    return np.array([
        np.random.uniform(min_coord, grid_size),
        np.random.uniform(min_coord, grid_size)
    ])

if __name__ == "__main__":
    # Test all curriculum levels
    for level in range(1, 2):
        try:
            env = generate_curriculum_environment(level, gen_fig=True)
            plt.title(f"Curriculum Level {level}")
            plt.show()
            print(f"Level {level} obstacles:\n{env['obstacles']}")
        except Exception as e:
            print(f"Error generating level {level}: {str(e)}")
