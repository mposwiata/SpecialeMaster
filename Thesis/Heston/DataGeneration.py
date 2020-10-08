import numpy as np
import itertools
import time

from Thesis.Heston import AndersenLake as al, HestonModel as hm
from Thesis.misc import VanillaOptions as vo

def calcImpVol(inputArray : np.array, optionList : np.array) -> np.ndarray:
    someModel = hm.HestonClass(inputArray[0], inputArray[1], inputArray[2], inputArray[3], inputArray[4], inputArray[5], inputArray[6])
    outputLenght = np.shape(optionList)[0]
    output = np.empty(outputLenght, dtype=np.float64)
    for i in range(np.shape(optionList)[0]):
        try:
            output[i] = al.Andersen_Lake(someModel, optionList[i])
        except:
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
maturity = np.linspace(start = 0.5, stop = 2, num = 2)

# strike
strike = np.linspace(start = 75, stop = 125, num = 2)

model_input = np.array(list(itertools.product(forward, vol, kappa, theta, epsilon, rho, rate))) # model parameter combinations
option_input = np.array(list(itertools.product(maturity, strike))) # different option combinations
someOptionList = np.array([])
for option in option_input:
    someOptionList = np.append(someOptionList, vo.EUCall(option[0], option[1]))

parallel_set = np.array_split(model_input, 4, axis=0)


#print(al.Andersen_Lake(someHestonModel, someCallOption))
