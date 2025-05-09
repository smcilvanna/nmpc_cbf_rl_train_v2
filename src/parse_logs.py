import numpy as np
import matplotlib.pyplot as plt
import glob

rew_values = []
actor_loss = []
critic_loss = []
folder_path = "train_data/old/train2"
for filename in sorted(glob.glob(f"{folder_path}/*.txt")):
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
                critic_loss.append(float(value))
                

data = np.array(rew_values)
dat2 = np.array(actor_loss)
dat3 = np.array(critic_loss)

fig, axs = plt.subplots(3, 1, figsize=(9, 3*3))  # Each subplot 1:3 aspect (3 tall, 9 wide)
axs[0].plot(data)
axs[0].set_ylabel("Mean Reward")
axs[1].plot(dat2)
axs[1].set_ylabel("Actor Loss")
axs[2].plot(dat3)
axs[2].set_ylabel("Critic Loss")
axs[2].set_xlabel("")
# scale = 4
# for ax in axs:
#     # Get the current x-ticks
#     ticks = ax.get_xticks()
#     # Scale the tick labels by multiplying by the scale factor
#     ax.set_xticklabels([str(int(tick * scale)) for tick in ticks])
plt.tight_layout()
plt.show()