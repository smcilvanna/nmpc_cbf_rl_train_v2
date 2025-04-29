# Observation Vector Definition for RL Training

## **Structure & Dimensions**
- **Total elements**: 64 (`1 + 3 + 20×3 = 64`)
- **Data type**: `numpy.float32`

---

## **1. MPC Performance**
**Position**: `[0]`  
**Description**: Execution time of last MPC calculation  
**Values**:
- `0.0` if unavailable
- Positive float representing seconds

---

## **2. Target Information**
**Position**: `[1:4]`  
**Components**:
| Index | Description                | Calculation                     |
|-------|----------------------------|---------------------------------|
| 1     | Euclidean distance to target | `‖target_pos[:2] - agent_pos[:2]‖` |
| 2     | Sine of bearing angle      | `sin(arctan2(target_y, target_x))` |
| 3     | Cosine of bearing angle    | `cos(arctan2(target_y, target_x))` |

**Notes**:
- Uses XY-plane projection (ignores Z-axis)
- Bearing relative to agent's current position

---

## **3. Obstacle Information**
**Position**: `[4:64]` (20 obstacles × 3 values each)  
**Per-Obstacle Structure**:
[adjusted_distance, sin(bearing_angle), cos(bearing_angle)]

**Calculations**:
1. **Adjusted Distance**:
- `0.55`: Agent's collision radius
- `obstacle[2]`: Obstacle's radius
2. **Bearing Components**:
- Same sine/cosine format as target bearing

**Order**: Obstacles are processed in `self.map['obstacles']` order

---

## **Visual Summary**
| Section           | Elements | Indices   | Description                         |
|-------------------|----------|-----------|-------------------------------------|
| MPC Performance   | 1        | [0]       | Control system timing metric        |
| Target Info       | 3        | [1-3]     | Relative target position encoding   |
| Obstacle Info     | 60       | [4-63]    | 20×3 obstacle proximity signals     |
