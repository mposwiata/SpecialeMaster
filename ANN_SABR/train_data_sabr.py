import numpy as np
import time
import itertools
import pandas
from scipy.stats import norm
from Thesis.Functions import SABR_ANN as sa
from Thesis.Functions import hagan

# Maturity
mat_1Y = np.linspace(start = 1/360, stop = 1, num = 100)
mat_2Y = mat_1Y + 1

# Alpha, starting vol
alpha = np.linspace(start = 0.05, stop = 0.5, num = 10)

# Rho
rho = np.linspace(start = -0.9, stop = 0.9, num = 10)

# Eta
eta_s = np.array([-3.5, 3.5])
eta_0 = 1.5

# Nu
Ts = 1/12
nu = np.empty([np.shape(mat_1Y)[0],10])
i = 0
for mat in mat_1Y:
    nu[i,:] = sa.nu_par(mat, Ts)
    i += 1

nu_dict = dict(zip(mat_1Y, nu)) # hashmap for matyrities and corresponding vol-vol's

# Creating the input array
input_array = np.array(list(itertools.product(alpha, rho, nu_dict[mat_1Y[0]])))
mat_array = np.repeat(mat_1Y[0],np.shape(input_array)[0])
input_array = np.column_stack((mat_array, input_array))
for mat in mat_1Y[1:]:
    mat_input = np.array(list(itertools.product(alpha, rho, nu_dict[mat])))
    mat_input_array = np.repeat(mat, np.shape(mat_input)[0])
    input_array = np.vstack((input_array, np.column_stack((mat_input_array, mat_input))))
empty_strike = np.empty([100000, 10])
input_array = np.column_stack((input_array, empty_strike))
i = 0
for row in input_array:
    strike_array = sa.strike_par(row[0], row[1], row[2], row[3])
    input_array[i, 4:] = np.linspace(start = strike_array[0], stop = strike_array[1], num = 10)
    i += 1
# Creating the output array
output_array = np.empty((np.shape(input_array)[0], 10))
i = 0

for row in input_array:
    j = 0
    for strike in input_array[i, 4:]:
        output_array[i, j] = hagan.hagan_sigma_b(row[0], row[1], row[2], row[3], strike)
        j += 1
    i += 1

random_indices_train = np.random.choice(100000, size = 150000, replace = True) # extra 150k training
random_indices_test = np.random.choice(100000, size = 50000, replace = True) # 50k random for test
train_array = np.concatenate((input_array, input_array[random_indices_train, :]), 0)
test_array = input_array[random_indices_test, :]
train_output_array = np.concatenate((output_array, output_array[random_indices_train, :]), 0)
test_output_array = output_array[random_indices_test, :]

np.savetxt("train_input_sabr.csv", train_array, delimiter=",")
np.savetxt("train_output_sabr_approx.csv", train_output_array, delimiter=",")
np.savetxt("test_input_sabr.csv", test_array, delimiter=",")
np.savetxt("test_output_sabr_approx.csv", test_output_array, delimiter=",")