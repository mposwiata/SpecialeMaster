import numpy as np
import itertools
import time
from multiprocess import Pool

from Thesis.Heston import AndersenLake as al, HestonModel as hm
from Thesis.misc import VanillaOptions as vo

def calcImpVol(inputArray : np.array, optionList : np.array) -> np.ndarray:
    someModel = hm.HestonClass(inputArray[0], inputArray[1], inputArray[2], inputArray[3], inputArray[4], inputArray[5], inputArray[6])
    outputLenght = np.shape(optionList)[0]
    output = np.empty(outputLenght, dtype=np.float64)
    
    for i in range(outputLenght):
        try:
            output[i] = someModel.impVol(al.Andersen_Lake(someModel, optionList[i]), optionList[i])
        except: #overflow in char function, set impvol to 0
            output[i] = 0

    return output

def impVolGenerator(inputArray : np.ndarray, optionList : np.array) -> np.ndarray:
    output_matrix = np.empty([np.shape(inputArray)[0], np.shape(optionList)[0]])
    i = 0
    for someInput in inputArray:
        output_matrix[i, :] = calcImpVol(someInput, optionList)
        i += 1
    
    return output_matrix

# Forward
forward = np.linspace(start = 75, stop = 125, num = 10)

# vol
vol = np.linspace(start = 0.01, stop = 0.1, num = 2)

# kappa
kappa = np.linspace(start = 0.1, stop = 2, num = 2)

# theta
theta = np.linspace(start = 0.01, stop = 0.1, num = 2)

# epsilon
epsilon = np.linspace(start = 0.1, stop = 2, num = 2)

# rho
rho = np.linspace(start = -0.99, stop = 0.99, num = 2)

# rate
rate = np.linspace(start = 0, stop = 0.2, num = 2)

# Maturity
maturity = np.linspace(start = 0.5, stop = 1, num = 2)

# strike
strike = np.linspace(start = 100, stop = 105, num = 2)

model_input = np.array(list(itertools.product(forward, vol, kappa, theta, epsilon, rho, rate))) # model parameter combinations
option_input = np.array(list(itertools.product(maturity, strike))) # different option combinations
someOptionList = np.array([])
for option in option_input:
    someOptionList = np.append(someOptionList, vo.EUCall(option[0], option[1]))

# generating data for neural net with model as input and grid as output
input1 = model_input
start = time.time()
# going parallel
pool = Pool(2)
parallel_set = np.array_split(model_input, 4, axis=0)
parallel_input = [
    [parallel_set[0], someOptionList],
    [parallel_set[1], someOptionList],
    [parallel_set[2], someOptionList],
    [parallel_set[3], someOptionList]
]
res = pool.starmap(impVolGenerator, parallel_input)
output1 = np.concatenate(res, axis = 0)
stop = time.time()
print("time: ", stop-start)
np.savetxt("hestonGridInput.csv", input1, delimiter=",")
np.savetxt("hestonGridOutput.csv", output1, delimiter=",")

# generating data for nn with all inputs and 1 output price
total_comb = np.shape(model_input)[0] * np.shape(output1)[1]
total_cols = np.shape(model_input)[1] + np.shape(option_input)[1]
total_options = np.shape(option_input)[0]
input2 = np.empty((total_comb, total_cols))
output2 = np.empty((total_comb, 1))
for i in range(np.shape(model_input)[0]):
    for j in range(total_options):
        input2[i*total_options+j, 0:np.shape(model_input)[1]] = model_input[i]
        input2[i*total_options+j, (np.shape(model_input)[1]) : total_cols] = option_input[j]
        output2[i*total_options+j] = output1[i, j]
np.savetxt("hestonSingleInput.csv", input2, delimiter=",")
np.savetxt("hestonSingleOutput.csv", output2, delimiter=",")

"""
# Data generation through parallel
start = time.time()
pool = Pool(2)
parallel_set = np.array_split(model_input, 4, axis=0)
parallel_input = [
    [parallel_set[0], someOptionList],
    [parallel_set[1], someOptionList],
    [parallel_set[2], someOptionList],
    [parallel_set[3], someOptionList]
]
res = pool.starmap(impVolGenerator, parallel_input)
result = np.concatenate(res, axis = 0)
print(np.shape(result))
stop = time.time()
print(stop-start)
"""