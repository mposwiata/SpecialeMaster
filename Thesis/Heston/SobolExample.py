import numpy as np
import sobol_seq as sobol
import matplotlib.pyplot as plt
import itertools

sobol_seq = sobol.i4_sobol_generate(2, 100)
lin_input = np.linspace(start = 0, stop = 1, num = 10)
lin_seq = np.array(list(itertools.product(lin_input, lin_input)))

fig = plt.figure()
sobol_ax = fig.add_subplot(121)
lin_ax = fig.add_subplot(122)
sobol_ax.scatter(sobol_seq[:,0], sobol_seq[:,1])
sobol_ax.set_title("Sobol sequence")
lin_ax.scatter(uniform_seq[:,0], lin_seq[:,1])
lin_ax.set_title("Uniform sequence")
plt.show()