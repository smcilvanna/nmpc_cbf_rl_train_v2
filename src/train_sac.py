import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from custom_env_horizon import MPCHorizonEnv

# Curriculum schedule
CURRICULUM_STAGES = [
    # {"level": 1, "steps": 2e6, "name": "basic"},
    {"level": 2, "steps": 5e5, "name": "gates1"},
    {"level": 2, "steps": 5e5, "name": "gates2"},
    {"level": 2, "steps": 5e5, "name": "gates3"},
    {"level": 2, "steps": 5e5, "name": "gates4"},
    {"level": 2, "steps": 5e5, "name": "gates5"}
    # {"level": 3, "steps": 5e5, "name": "complex"}
]

retrain = True
trainid = 2
retrainid = 5
SAVE_DIR = f"./train_data/train{trainid}/models{retrainid}/"
LOG_DIR =  f"./train_data/train{trainid}/logs{retrainid}/"
N_ENVS = 6  # Number of parallel environments


def train():
    if retrain:
        model = SAC.load('train_data/train2/models4/sac_cbf_3x-2-4-final.zip')
    else:
        model = None
    for stage in CURRICULUM_STAGES:
        # Create vectorized environments for current stage
        env = make_vec_env(
            lambda: MPCHorizonEnv(curriculum_level=stage["level"]),
            n_envs=N_ENVS,
            vec_env_cls=SubprocVecEnv,
            monitor_dir=LOG_DIR
        )

        if model is None:
            # Initialize new SAC model
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=100_000,
                learning_starts=5_000,
                batch_size=256,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
                verbose=1,
                # tensorboard_log=LOG_DIR
            )
        else:
            # Update environment for existing model
            model.set_env(env)

        # Create evaluation callback
        eval_callback = EvalCallback(
            env,
            best_model_save_path=SAVE_DIR,
            eval_freq=max(stage["steps"]//10, 1e4),
            deterministic=True,
            render=False
        )

        # Train for current stage
        model.learn(
            total_timesteps=int(stage["steps"]),
            # callback=eval_callback,
            # tb_log_name=f"sac_cbf_{stage['name']}",
            reset_num_timesteps=False
        )

        # Save stage checkpoint
        model.save(os.path.join(SAVE_DIR, f"sac_cbf_3x-{trainid}-{retrainid}-{stage['name']}"))

    # Save final model
    model.save(os.path.join(SAVE_DIR, f"sac_cbf_3x-{trainid}-{retrainid}-final"))

if __name__ == "__main__":
    train()