from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from custom_env import CustomSystemEnv
from action_persistence_wrapper import ActionPersistenceWrapper

# Configuration
NUM_ENVS = 1  # Number of parallel environments
TOTAL_TIMESTEPS = 10000 #1_000_000
SAVE_PATH = "./sac_temp/sac_custom_system"

def create_env():
    env = CustomSystemEnv()
    env = ActionPersistenceWrapper(env, persist_steps=10)
    return env

if __name__ == "__main__":
    # Create vectorized environments
    vec_env = make_vec_env(
        create_env, 
        n_envs=NUM_ENVS,
        vec_env_cls=SubprocVecEnv
    )

    # Initialize SAC agent
    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        # tensorboard_log="./tensorboard_logs/",
        buffer_size=1_000_000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=10,
        gradient_steps=10,
        policy_kwargs=dict(net_arch=[256, 256])
    )

    # Train the agent
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(SAVE_PATH)
    print(f"Training complete. Model saved to {SAVE_PATH}")
