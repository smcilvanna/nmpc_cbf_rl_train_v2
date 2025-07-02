# NMPC-CBF AGV RL Parameter Tuning

This repository contains the code for the work in chapter 5 of the thesis.  
There are separate branches within this repository for training the CBF parameter agent and the horizon length.


## Combined Agent Experiment Video
The trained SAC-CBF agent and the PPO horizon agent are tested on the Clearpath A200 Husky AGV.  
Without the horizon adjustment the agent can run into deadlock condition but with the RL adjustment it is able to navigate around the obstacles and lower the horizon to improve solve time performance when appropriate.  

