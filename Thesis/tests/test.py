import numpy as np
import time
import itertools
from scipy.stats import norm
from Thesis.Models import black_scholes as bs
from Thesis.Functions import monte_carlo as mc

strike_range = np.arange(80, 120, 5)
time_range = np.arange(0.05, 2.05, 0.05)
sigma_range = np.arange(0.05, 0.55, 0.05)
r_range = np.arange(0, 0.11, 0.01)
spot = 100

input_data = np.array(list(itertools.product(time_range, strike_range, sigma_range, r_range)))

rows = np.shape(input_data)[0]

ekstra_data = np.full((rows, 1), spot)

first_input = np.concatenate((ekstra_data, input_data), 1)

random_indices = np.random.choice(rows, size = rows, replace = False)

final_input = np.concatenate((first_input, first_input[random_indices, :]), 0)

output = np.zeros(np.shape(final_input)[0])

i = 0
for row in final_input:
    output[i] = mc.monteCarloBS(row[0], row[1], row[2], row[3], row[4], 100000)
    i += 1

np.savetxt("input.csv", final_input, delimiter = ",")
np.savetxt("output.csv", output)