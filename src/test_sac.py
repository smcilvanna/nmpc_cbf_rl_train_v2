from custom_env import CustomSystemEnv
from action_persistence_wrapper import ActionPersistenceWrapper

env = ActionPersistenceWrapper(CustomSystemEnv())
obs, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    print(f"Step: {_}, Obs shape: {len(obs)}, Reward: {reward}")
    if done:
        break