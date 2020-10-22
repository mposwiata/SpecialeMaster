import numpy as np
import itertools
import time
import sys
import os
sys.path.append(os.getcwd()) # added for calc server support
from multiprocess import Pool, cpu_count

from Thesis.Heston import AndersenLake as al, HestonModel as hm
from Thesis.misc import VanillaOptions as vo

def calcImpVol(inputArray : np.array, optionList : np.array) -> (np.array, np.array):
    someModel = hm.HestonClass(inputArray[0], inputArray[1], inputArray[2], inputArray[3], inputArray[4], inputArray[5], inputArray[6])
    outputLenght = np.shape(optionList)[0]
    output_price = np.empty(outputLenght, dtype=np.float64)
    output_impVol = np.empty(outputLenght, dtype=np.float64)
    
    for i in range(outputLenght):
        try:
            output_price[i] = al.Andersen_Lake(someModel, optionList[i])
            output_impVol[i] = someModel.impVol(output_price[i], optionList[i])
        except: #overflow in char function, set impvol to 0
            output_price[i] = 0
            output_impVol[i] = 0

    return output_price, output_impVol

def impVolGenerator(inputArray : np.ndarray, optionList : np.array) -> (np.ndarray, np.ndarray):
    output_price_matrix = np.empty([np.shape(inputArray)[0], np.shape(optionList)[0]])
    output_impVol_matrix = np.empty([np.shape(inputArray)[0], np.shape(optionList)[0]])
    i = 0
    for someInput in inputArray:
        output_price_matrix[i, :], output_impVol_matrix[i, :] = calcImpVol(someInput, optionList)
        i += 1
    
    return output_price_matrix, output_impVol_matrix

def modelInputGenerator() -> np.ndarray:
    # Forward
    forward = np.linspace(start = 75, stop = 125, num = 10)

    # vol
    vol = np.linspace(start = 0.01, stop = 0.2, num = 5)

    # kappa
    kappa = np.linspace(start = 0.1, stop = 2, num = 5)

    # theta
    theta = np.linspace(start = 0.01, stop = 0.2, num = 5)

    # epsilon
    epsilon = np.linspace(start = 0.1, stop = 2, num = 5)

    # rho
    rho = np.linspace(start = -0.99, stop = 0.99, num = 10)

    # rate
    rate = np.linspace(start = 0, stop = 0.2, num = 5)

    return np.array(list(itertools.product(forward, vol, kappa, theta, epsilon, rho, rate))) # model parameter combinations

def optionInputGenerator() -> np.ndarray:
    # Maturity
    maturity = np.linspace(start = 0.01, stop = 2, num = 5)

    # strike
    strike = np.linspace(start = 75, stop = 125, num = 5)

    return np.array(list(itertools.product(maturity, strike)))



if __name__ == "__main__":
    model_input = modelInputGenerator()

    option_input = optionInputGenerator() # different option combinations
    someOptionList = np.array([])
    for option in option_input:
        someOptionList = np.append(someOptionList, vo.EUCall(option[0], option[1]))

    # generating data for neural net with model as input and grid as output
    gridInput = model_input

    # going parallel
    cpu_cores = cpu_count()
    parallel_set = np.array_split(model_input, cpu_cores, axis=0)
    parallel_list = []

    # generating list of datasets for parallel
    for i in range(cpu_cores):
        parallel_list.append((parallel_set[i], someOptionList))

    # parallel
    pool = Pool(cpu_cores)
    res = pool.starmap(impVolGenerator, parallel_list)
    res = np.concatenate(res, axis = 1)
    price_output = res[0]
    impVol_output = res[1]

    # saving grid datasets
    np.savetxt("Data/hestonGridInput.csv", gridInput, delimiter=",")
    np.savetxt("Data/hestonGridPrice.csv", price_output, delimiter=",")
    np.savetxt("Data/hestonGridImpVol.csv", impVol_output, delimiter=",")

    """
    # Generating single outputs
    total_comb = np.shape(model_input)[0] * np.shape(option_input)[0]
    total_cols = np.shape(model_input)[1] + np.shape(option_input)[1]
    total_options = np.shape(option_input)[0]
    singleInput = np.empty((total_comb, total_cols))
    singlePrice_output = np.empty((total_comb, 1))
    singleImpVol_output = np.empty((total_comb, 1))
    for i in range(np.shape(model_input)[0]):
        for j in range(total_options):
            singleInput[i*total_options+j, 0:np.shape(model_input)[1]] = model_input[i]
            singleInput[i*total_options+j, (np.shape(model_input)[1]) : total_cols] = option_input[j]
            singlePrice_output[i*total_options+j] = price_output[i, j]
            singleImpVol_output[i*total_options+j] = impVol_output[i, j]

    # saving dataset2
    np.savetxt("Data/hestonSingleInput.csv", singleInput, delimiter=",")
    np.savetxt("Data/hestonSinglePrice.csv", singlePrice_output, delimiter=",")
    np.savetxt("Data/hestonSingleImpVol.csv", impVol_output, delimiter=",")
    """