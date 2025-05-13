import numpy as np
import matplotlib.pyplot as plt
import glob

rew_values = []
actor_loss = []
critic_loss = []
episode = []
for filename in sorted(glob.glob("./train_data/train2/*.log")):
    print(filename)
    with open(filename, "r") as f:
        for line in f:
            if "ep_rew_mean" in line:
                value = line.strip().split("|")[-2].strip()
                rew_values.append(float(value))
            if "actor_loss" in line:
                value = line.strip().split("|")[-2].strip()
                actor_loss.append(float(value))
            if "critic_loss" in line:
                value = line.strip().split("|")[-2].strip()
                critic_loss.append(min(float(value),10e3))
            if "episodes" in line:
                value = line.strip().split("|")[-2].strip()
                episode.append(float(value))
                
data = np.array(rew_values[7:])
dat2 = np.array(actor_loss)
dat3 = np.array(critic_loss)
datx = np.array(episode[7:])

fig, axs = plt.subplots(3, 1, figsize=(9, 3*3))  # Each subplot 1:3 aspect (3 tall, 9 wide)
axs[0].plot(datx, data, c="red", lw=3)
axs[0].set_ylabel("Mean Reward")
axs[0].set_ylim(-700,700)
axs[1].plot(datx, dat2, c="green",lw=1)
axs[1].set_ylabel("Actor Loss")
axs[2].plot(datx, dat3, c="blue",lw=1)
axs[2].set_ylabel("Critic Loss")
axs[2].set_xlabel("Training Episode")
axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)
fig.suptitle("CBF Step Adjust SAC Training")
# scale = 4
# for ax in axs:
#     # Get the current x-ticks
#     ticks = ax.get_xticks()
#     # Scale the tick labels by multiplying by the scale factor
#     ax.set_xticklabels([str(int(tick * scale)) for tick in ticks])
plt.tight_layout()
plt.show()