import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(start = -5, stop = 5, num=1000)

def lr_schedule(epoch, rate):
    lower_lr = 1e-4
    upper_lr = lower_lr * 100
    no_epochs = 100
    peak_epoch = 45
    if epoch <= peak_epoch:
        lr = lower_lr + epoch / peak_epoch * (upper_lr - lower_lr)
    elif peak_epoch < epoch < peak_epoch * 2:
        lr = upper_lr - (epoch - peak_epoch) / peak_epoch * (upper_lr - lower_lr)
    else:
        lr = lower_lr * (1 - (epoch - 2 * peak_epoch) / (no_epochs - 2 * peak_epoch)) * (1 - 1 / 10)

    return lr

epochs = np.linspace(start = 1, stop=100, num=100)

lr_values = []
for some_epoch in epochs:
    lr_values.append(lr_schedule(some_epoch-1, 0))


fig = plt.figure(figsize=(10, 10), dpi = 200)
ax = fig.add_subplot(111)
ax.plot(epochs, lr_values)
plt.savefig("one_cycle_lr.png")
