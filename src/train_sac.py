import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from custom_env_horizon import MPCHorizonEnv

# Hyperparameters
CURRICULUM_LEVEL = 1
TOTAL_TIMESTEPS = 500_000
SAVE_DIR = "./models/"
LOG_DIR = "./logs/"

# Create environment
env = MPCHorizonEnv(curriculum_level=CURRICULUM_LEVEL)
env = Monitor(env, LOG_DIR)  # Log metrics

# Policy Architecture (Customize for obstacle awareness)
policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], qf=[256, 256]),
    features_extractor_kwargs=dict(features_dim=32),  # Optional: Add custom feature extractor
)

# Initialize SAC agent
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=5_000,
    batch_size=256,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=LOG_DIR
)

# Callbacks
eval_callback = EvalCallback(
    env,
    best_model_save_path=SAVE_DIR,
    eval_freq=10_000,
    deterministic=True,
    render=False,
)

# Train
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
    tb_log_name="sac_cbf_tuning"
)

# Save final model
model.save(os.path.join(SAVE_DIR, "sac_cbf_final"))
