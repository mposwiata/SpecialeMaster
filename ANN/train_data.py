import numpy as np
import time
import itertools
import pandas
from scipy.stats import norm
from Thesis.Models import black_scholes as bs
from Thesis.Functions import monte_carlo as mc
from Thesis.Functions import mc_numba as mcn

spot = np.arange(10, 200, 2)
strike = np.random.random(len(spot)) + 0.5 #random variable on (0,1)
r = 0.05
mat = np.arange(0.1, 1, 0.10)
sigma = np.arange(0.1, 0.6, 0.05)

data = np.array(list(itertools.product(spot, (strike * spot), mat, sigma)))

output = np.empty([np.shape(data)[0],1])


i = 0
for row in data:
    output[i] = bs.BlackScholesFormula(row[0], row[1], row[2], row[3], r)
    i += 1

np.savetxt("input.csv", data, delimiter=",")
np.savetxt("output.csv", output)
 
"""
input_array = np.array(list(itertools.product(mat_range, strike_range, sigma_range, r_range))) #array of all combinations

input_len = np.shape(input_array)[0] #how many rows

spot_col = np.full((input_len, 1), spot_value) #spot column

input_base = np.concatenate((input_array, spot_col), 1)

random_indices = np.random.choice(input_len, size = input_len, replace = False) #taking 50-50 fixed / random 

final_input = np.concatenate((input_base, input_base[random_indices, :]), 0)

output = np.zeros(np.shape(final_input)[0])

i = 0
for row in final_input:
    output[i] = mcn.monteCarloBS(row[0], row[1], row[2], row[3], row[4], 100000)
    i += 1

#create pandas
panda_input = pandas.DataFrame(final_input, columns=['mat', 'strike', 'sigma', 'r', 'spot'])
panda_output = pandas.DataFrame(output, columns=['price'])

#save as csv for later use
panda_input.to_csv("input.csv")
panda_output.to_csv("output.csv")
"""