import numpy as np
import time
import matplotlib.pyplot as plt

from Thesis.Heston import AndersenLake as al, MonteCarlo as mc, HestonModel as hm
from Thesis.misc import VanillaOptions as vo

some_model = hm.HestonClass(100, 0.04, 2, 0.04, 0.5, -0.7, 0)
some_option = vo.EUCall(1, 120)

some_model_hard = hm.HestonClass(100, 0.01, 0.1, 0.01, 2, 0.8, 0)

price = al.Andersen_Lake(some_model, some_option)
price_hard = al.Andersen_Lake(some_model_hard, some_option)
paths1 = np.linspace(start = 1, stop = 1000, num=50).astype(int)
paths2 = np.linspace(start = 1100, stop = 100000, num=100).astype(int)
paths = np.concatenate((paths1, paths2))
mc_price = []
mc_price_hard = []
for some_path in paths:
    mc_price.append(mc.Heston_monte_carlo(some_model, some_option, some_path))
    mc_price_hard.append(mc.Heston_monte_carlo(some_model_hard, some_option, some_path))

fig = plt.figure(figsize=(10, 10), dpi = 200)
ax = fig.add_subplot(111)
ax.plot(paths, mc_price, label="Monte Carlo, easy case")
ax.plot(paths, mc_price_hard, label="Monte Carlo, hard case")
ax.plot(paths, np.repeat(price, len(paths)), color = 'k', label="Andersen Lake, easy case")
ax.plot(paths, np.repeat(price_hard, len(paths)), color = 'y', label="Andersen Lake, hard case")
ax.set_ylabel("Price", rotation="horizontal", labelpad=15)
ax.set_xlabel("Paths")
ax.set_title("Monte Carlo vs. Andersen Lake")
ax.legend(loc="upper right")
plt.savefig("mc_al_compare.png")

al_start = time.time()
al.Andersen_Lake(some_model, some_option)
al_stop = time.time()

print("AL time: ", (al_stop - al_start))

test_paths = [1, 10, 100, 1000, 10000, 25000, 50000, 75000, 100000]
for some_path in test_paths:
    mc_start = time.time()
    mc.Heston_monte_carlo(some_model, some_option, some_path)
    mc_stop = time.time()
    print("time for "+str(some_path)+": ", (mc_stop-mc_start))