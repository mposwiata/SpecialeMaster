import numpy as np
import time
import itertools
import sys
import os
from scipy.stats import norm
sys.path.append(os.getcwd()) #added cause windows wont add to PYTHONPATH

from Thesis.Functions import monte_carlo as mc
from Thesis.Functions import mc_numba as mc_n


t11 = time.time()
for i in range(1, 100):
    mc.monteCarloBS(100, 1, 100, 0.2, 0.05, 1000000)
t12 = time.time()

mc_n.monteCarloBS(100, 1, 100, 0.2, 0.05, 1)

t21 = time.time()
for i in range(1, 100):
    mc_n.monteCarloBS(100, 1, 100, 0.2, 0.05, 1000000)
t22 = time.time()
print("Time1: ", t12-t11)
print("Time2: ", t22-t21)