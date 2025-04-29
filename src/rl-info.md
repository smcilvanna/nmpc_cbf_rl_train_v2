# Reinforcement Learning For MPC-CBF

## Observations Vector Definition 

Number of elements [93]

#                      0     1      2         3          4        5         6    7
  state (8)       : [x-pos y-pos  x-tgt-n   y-tgt-n   sin(yaw) cos(yaw)     v    w ]
#                      8+o      9+o        10+o      11+o    (o =obsIndex*4)
  obsObv (80)     : [dist, sin(angle), cos(angle), radius]
#                       88    89           90          91
  targetInfo (4)  : [ dist, sin(angle), cos(angle) progress]
#                       92
  mpcTime  (1)    : [ mpcTime ]  