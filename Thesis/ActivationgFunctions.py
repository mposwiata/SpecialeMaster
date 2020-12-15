import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(start = -5, stop = 5, num=1000)

def softplus(x : float) -> float:
    return np.log(np.exp(x)+1)

def tanh(x : float) -> float:
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

softplus_values = softplus(x)
tanh_values = tanh(x)


fig = plt.figure(figsize=(10, 10), dpi = 200)
ax = fig.add_subplot(111)
ax.plot(x, softplus_values, label="softplus")
ax.tick_params(axis = "both", labelsize=15)
ax.plot(x, tanh_values, label="tanh")
ax.legend(loc="upper left", prop={'size': 20})
plt.savefig("activation_functions_base.png")
