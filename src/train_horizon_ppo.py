from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from custom_env_horizon import MPCHorizonEnv

# Curriculum schedule
CURRICULUM_STAGES = [
    {"level": 1, "steps": 2e5, "name": "basic"},
    {"level": 2, "steps": 3e5, "name": "gates"},
    {"level": 3, "steps": 5e5, "name": "complex"}
]

def train():
    model = None
    for stage in CURRICULUM_STAGES:
        # Create vectorized environments
        env = make_vec_env(
            lambda: MPCHorizonEnv(curriculum_level=stage["level"]), 
            n_envs=4,
            vec_env_cls=SubprocVecEnv
        )
        
        if model is None:
            # Initialize new model
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                ent_coef=0.01,
                policy_kwargs=dict(net_arch=[256, 256])
            )
        else:
            # Update environment for existing model
            model.set_env(env)
        
        # Train for current stage
        model.learn(total_timesteps=int(stage["steps"]))
        
        # Save checkpoint
        model.save(f"ppo_mpc_horizon_{stage['name']}")

if __name__ == "__main__":
    train()