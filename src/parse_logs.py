import numpy as np
import matplotlib.pyplot as plt
import glob

rew_values = []
actor_loss = []
critic_loss = []
episode = []
entropy = []
folder_path = "train_data/train5"
for filename in sorted(glob.glob(f"{folder_path}/*.txt")):
    print(filename)
    with open(filename, "r") as f:
        for line in f:
            if "iterations" in line:
                ep = line.strip().split("|")[-2].strip()
                episode.append(float(ep))
            if "ep_rew_mean" in line:
                value = line.strip().split("|")[-2].strip()
                rew_values.append(float(value))
            if "policy_gradient_loss" in line:
                value = line.strip().split("|")[-2].strip()
                actor_loss.append(float(value))
            if "value_loss" in line:
                value = line.strip().split("|")[-2].strip()
                critic_loss.append(float(value))
            if "entropy_loss" in line:
                value = line.strip().split("|")[-2].strip()
                entropy.append(float(value))
                

data = np.array(rew_values)
dat2 = np.array(actor_loss)
dat3 = np.array(critic_loss)
datx = np.array(episode)
dat4 = np.array(entropy)


print(data.shape, dat2.shape, dat3.shape, datx.shape, dat4.shape)

fig, axs = plt.subplots(4, 1, figsize=(9, 3*3))  # Each subplot 1:3 aspect (3 tall, 9 wide)
axs[0].plot(data, c="red", lw=3)
axs[0].set_ylabel("Mean Reward")
axs[1].plot(dat2, c="green",lw=1)
axs[1].set_ylabel("Actor Loss")
axs[2].plot(dat3, c="blue",lw=1)
axs[2].set_ylabel("Critic Loss")
axs[3].plot(dat4, c="grey", lw=1)
axs[3].set_ylabel("Entropy Loss")
axs[3].set_xlabel("Policy Update Step")
fig.suptitle("Horizon Step Adjust PPO Training")
# scale = 4
# for ax in axs:
#     # Get the current x-ticks
#     ticks = ax.get_xticks()
#     # Scale the tick labels by multiplying by the scale factor
#     ax.set_xticklabels([str(int(tick * scale)) for tick in ticks])
plt.tight_layout()
plt.show()