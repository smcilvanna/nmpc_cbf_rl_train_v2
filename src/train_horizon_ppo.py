from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from custom_env_horizon import MPCHorizonEnv, ActionPersistenceWrapper
from stable_baselines3.common.callbacks import BaseCallback

class CustomLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.horizon_lengths = []
        self.mpc_times = []
        self.collisions = []

    def _on_step(self) -> bool:
        # Extract metrics from all environments
        for env_idx in range(len(self.locals["infos"])):
            info = self.locals["infos"][env_idx]
            self.logger.record(f"env{env_idx}/mpc_time", info["mpc_time"])
            self.logger.record(f"env{env_idx}/horizon", info["horizon"])
            # self.logger.record(f"env{env_idx}/collision", float(info["collision"]))
            # self.logger.record(f"env{env_idx}/target_distance", info["target_distance"])
        return True
    
# Curriculum schedule
CURRICULUM_STAGES = [
    # {"level": 1, "steps": 2e5, "name": "basic"},
    {"level": 2, "steps": 5e5, "name": "med5"},
    {"level": 2, "steps": 5e5, "name": "med6"},
    {"level": 2, "steps": 5e5, "name": "med7"},
    # {"level": 3, "steps": 5e5, "name": "complex"}
]

retrain = True
train_id = 5
retrain_id = 1


def train():
    if retrain:
        model = PPO.load(f"ppo_mpc_horizon_ks_{train_id}-{retrain_id}_med4")
    else:
        model = None
    for stage in CURRICULUM_STAGES:
        # Create vectorized environments with ActionPersistenceWrapper
        env = make_vec_env(
            lambda: ActionPersistenceWrapper(MPCHorizonEnv(curriculum_level=stage["level"])), 
            n_envs=12,
            vec_env_cls=SubprocVecEnv
        )
        
        if model is None:
            # Initialize new model
            model = PPO(
                "MlpPolicy",
                env,
                # device="cpu",
                verbose=1,
                # tensorboard_log="./ppo_mpc_tensorboard/",  # Enable TensorBoard logging
                learning_rate=3e-4,
                n_steps=512,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                ent_coef=0.1,  # increased from 0.01
                policy_kwargs=dict(net_arch=[256, 256])
            )
        else:
            # Update environment for existing model
            model.set_env(env)
        
         # Train with custom callback
        model.learn(
            total_timesteps=int(stage["steps"]),
            # callback=CustomLoggingCallback(),  # Add callback
            # tb_log_name=f"curriculum_{stage['name']}"  # Unique log name per stage
        )
        
        # Save checkpoint
        model.save(f"ppo_mpc_horizon_ks_{train_id}-{retrain_id}_{stage['name']}")

if __name__ == "__main__":
    train()
