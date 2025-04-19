import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass
from typing import Optional

@dataclass
class MapLimits:
    x: list
    y: list

def generate_curriculum_environment(curriculum_level: int, gen_fig: bool = False) -> dict:
    """Generate a random environment with curriculum-based obstacle placement."""
    # ===== Curriculum Presets =====
    presets = {
        1: {'num_circles': 0, 'min_spacing': 10, 'grid_size': 10, 'target_path_obs': 0},
        2: {'num_circles': 3, 'min_spacing': 5, 'grid_size': 15, 'target_path_obs': 1},
        3: {'num_circles': 4, 'min_spacing': 4, 'grid_size': 20, 'target_path_obs': 2},
        4: {'num_circles': 6, 'min_spacing': 2, 'grid_size': 25, 'target_path_obs': 3},
        5: {'num_circles': 6, 'min_spacing': 1.5, 'grid_size': 30, 'target_path_obs': 3}
    }
    params = presets[curriculum_level]
    
    # Initialize parameters
    target_pos_normal = np.random.rand(2)
    mpc_req_obs = 6
    veh_rad = 1.0
    radii_set = np.arange(0.1, 10.1, 0.1)
    out = {
        'obs_in_path': -1,
        'c_level': curriculum_level,
        'mpc_req_obs': mpc_req_obs
    }
    c5cnt = 0

    while out['obs_in_path'] < params['target_path_obs']:
        # Initialize target position
        target_pos = np.array([params['grid_size'], params['grid_size']])
        if c5cnt < 2000 and curriculum_level != 5:
            idx = np.random.randint(2)
            target_pos[idx] = target_pos_normal[idx] * params['grid_size']
        
        # Initialize circles with vehicle positions
        circles = np.zeros((params['num_circles'] + 2, 3))
        circles[0] = [0, 0, veh_rad]
        circles[1] = [target_pos[0], target_pos[1], veh_rad]
        current_count = 0

        # Place circles with collision checking
        while current_count < params['num_circles']:
            x = np.random.rand() * params['grid_size']
            y = np.random.rand() * params['grid_size']
            radius = np.random.choice(radii_set)
            
            valid = True
            for i in range(current_count + 2):
                existing = circles[i]
                distance = np.linalg.norm([x-existing[0], y-existing[1]])
                min_required = radius + existing[2] + params['min_spacing']
                
                if distance < min_required:
                    valid = False
                    break

            if valid:
                current_count += 1
                circles[current_count+1] = [x, y, radius]

        # Calculate coverage
        total_area = params['grid_size'] ** 2
        covered_area = np.sum(np.pi * circles[2:, 2] ** 2)
        coverage = (covered_area / total_area) * 100
        
        # Check obstacles in path
        obstacles = circles[2:]
        out['obstacles'] = obstacles
        out['coverage'] = coverage
        
        if len(obstacles) > 0:
            circle_path = [line_intersects_circle([0, 0], target_pos, obs) for obs in obstacles]
            out['obs_in_path'] = np.sum(circle_path)
        else:
            out['obs_in_path'] = 0

        # Pad obstacles if needed
        while len(out['obstacles']) < mpc_req_obs:
            out['obstacles'] = np.vstack([out['obstacles'], [500, 500, 0.1]])

        # Curriculum level 5 special handling
        if curriculum_level == 5 and coverage < 60:
            out['obs_in_path'] = -1
            c5cnt += 1

    # Finalize output
    out['target_pos'] = target_pos
    out['map_limits'] = get_map_limits(out['obstacles'], curriculum_level, target_pos)
    
    # Generate figure if requested
    if gen_fig:
        fig, ax = plt.subplots()
        for obs in out['obstacles']:
            ax.add_patch(Circle(obs[:2], obs[2], fill=False, color='red'))
        ax.plot(target_pos[0], target_pos[1], 'gx', markersize=10)
        ax.set_xlim(out['map_limits'].x)
        ax.set_ylim(out['map_limits'].y)
        ax.set_aspect('equal')
        out['fig'] = fig
    else:
        out['fig'] = None

    return out

def get_map_limits(obstacles: np.ndarray, curriculum_level: int, target: np.ndarray) -> MapLimits:
    """Calculate map limits based on obstacles and target position."""
    if curriculum_level == 1:
        return MapLimits(x=[0, 20], y=[0, 20])
    
    xlims = [-0.55]
    ylims = [-0.55]
    for obs in obstacles:
        if obs[0] > 120:
            break
        xlims.extend([obs[0] - obs[2], obs[0] + obs[2]])
        ylims.extend([obs[1] - obs[2], obs[1] + obs[2]])
    
    map_limits = MapLimits(
        x=[min(xlims), max(xlims)] if xlims else [0, target[0]],
        y=[min(ylims), max(ylims)] if ylims else [0, target[1]]
    )
    
    # Extend limits to include target
    map_limits.x[1] = max(map_limits.x[1], target[0])
    map_limits.y[1] = max(map_limits.y[1], target[1])
    map_limits.x[1] = min(map_limits.x[1],map_limits.y[1])
    map_limits.y[1] = map_limits.x[1]
    map_limits.x[0] = min(map_limits.x[0],map_limits.y[0])
    map_limits.y[0] = map_limits.x[0]
    return map_limits

def line_intersects_circle(A: list, B: list, circle: np.ndarray) -> bool:
    """Check if line segment AB intersects with a circle."""
    C = circle[:2]
    radius = circle[2]
    AB = np.array(B) - np.array(A)
    AC = np.array(C) - np.array(A)
    
    t = np.dot(AC, AB) / np.dot(AB, AB)
    t_clamped = np.clip(t, 0, 1)
    D = A + t_clamped * AB
    
    distance = np.linalg.norm(D - C)
    return (distance <= radius) or \
           (np.linalg.norm(A - C) <= radius) or \
           (np.linalg.norm(B - C) <= radius)

if __name__ == "__main__":
    # Generate level 3 environment with figure
    env = generate_curriculum_environment(2, gen_fig=True)
    plt.show()

    # Access environment parameters
    print(f"Target position: {env['target_pos']}")
    print(f"Obstacles: {env['obstacles']}")
    print(f"Coverage: {env['coverage']:.1f}%")
