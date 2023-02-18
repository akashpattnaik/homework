"""
Plot the learning rate over time.
Adapted from https://gist.github.com/hatzel/8965ebf1231e3cf3c67caecbbb028dec
"""
# %%
from scheduler import CustomLRScheduler
import matplotlib.pyplot as plt
from config import CONFIG

from model import MiniCNN
import numpy as np

STEPS = 1563 * 5
# Create the model:
model = MiniCNN(num_channels=3)
# Create the optimizer:
optimizer = CONFIG.optimizer_factory(model)

scheduler = CustomLRScheduler(optimizer, **CONFIG.lrs_kwargs)

lrs = []
for _ in range(STEPS):
    optimizer.step()
    lrs.append(scheduler.get_lr())
    scheduler.step()

plt.plot(lrs)
plt.show()
# %%
plt.plot(np.abs(np.diff(np.array(lrs).squeeze())))
# %%
