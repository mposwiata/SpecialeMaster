import numpy as np
import time
import itertools
import pandas
from scipy.stats import norm
from Thesis.Models import black_scholes as bs
from Thesis.Functions import monte_carlo as mc
from Thesis.Functions import mc_numba as mcn

strike_range = np.arange(80, 120, 5)
mat_range = np.arange(0.05, 2.05, 0.05)
sigma_range = np.arange(0.05, 0.55, 0.05)
r_range = np.arange(0, 0.11, 0.01)
spot_value = 100

input_data_panda = pandas.DataFrame(list(itertools.product(mat_range, strike_range, sigma_range, r_range)), columns=['Mat', 'Strike', 'sigma', 'r'])