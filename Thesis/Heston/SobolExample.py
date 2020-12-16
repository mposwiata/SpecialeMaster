import numpy as np
import sobol_seq as sobol
import matplotlib.pyplot as plt
import itertools

sobol_seq = sobol.i4_sobol_generate(2, 256)
lin_input = np.linspace(start = 0, stop = 1, num = 16)
lin_seq = np.array(list(itertools.product(lin_input, lin_input)))
random_seq = np.random.uniform(size = (256, 2))

fig = plt.figure(figsize=(30, 10), dpi = 200)
sobol_ax = plt.subplot(131)
grid_ax = plt.subplot(132)
random_ax = plt.subplot(133)
sobol_ax.scatter(sobol_seq[:,0], sobol_seq[:,1])
grid_ax.scatter(lin_seq[:,0], lin_seq[:,1])
random_ax.scatter(random_seq[:, 0], random_seq[:, 1])

sobol_ax.tick_params(axis = "both", labelsize=15)
grid_ax.tick_params(axis = "both", labelsize=15)
random_ax.tick_params(axis = "both", labelsize=15)
sobol_ax.set_title("Sobol sequence", fontsize = 20)
grid_ax.set_title("Grid sequence", fontsize = 20)
random_ax.set_title("Uniform sequence", fontsize = 20)
fig.subplots_adjust(top=0.9, left=0.1, right=0.95, bottom=0.15)
plt.savefig("Sobol_grid.png")
plt.close()
