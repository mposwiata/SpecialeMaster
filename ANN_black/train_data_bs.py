import numpy as np
import time
import itertools
import pandas
from scipy.stats import norm
from Thesis.Models import black_scholes as bs
from Thesis.Functions import monte_carlo as mc
from Thesis.Functions import mc_numba as mcn

spot = np.linspace(start = 50, stop = 150, num = 20)
strike = np.linspace(start = 50, stop = 150, num = 20)
r = 0.05
mat = np.linspace(start = 0.1, stop = 2, num = 20)
sigma = np.linspace(0.05, 0.75, num = 15)

data = np.array(list(itertools.product(spot, strike, mat, sigma)))

output = np.empty([np.shape(data)[0],1])

random_indices_train = np.random.choice(20*20*20*15, size = 150000, replace = True) # extra 150k training
random_indices_test = np.random.choice(20*20*20*15, size = 50000, replace = True) # 50k random for test
train_array = np.concatenate((data, data[random_indices_train, :]), 0)
test_array = data[random_indices_test, :]
train_output_array = np.empty([np.shape(train_array)[0], 1])
test_output_array = np.empty([np.shape(test_array)[0], 1])

i = 0
for row in train_array:
    train_output_array[i] = mcn.monteCarloBS(row[0], row[1], row[2], row[3], r, 50000)
    i += 1

i = 0
for row in test_array:
    test_output_array[i] = mcn.monteCarloBS(row[0], row[1], row[2], row[3], r, 50000)
    i += 1

np.savetxt("train_input_bs_sim.csv", train_array, delimiter=",")
np.savetxt("train_output_bs_sim.csv", train_output_array, delimiter=",")
np.savetxt("test_input_bs_sim.csv", test_array, delimiter=",")
np.savetxt("test_output_bs_sim.csv", test_output_array, delimiter=",")
