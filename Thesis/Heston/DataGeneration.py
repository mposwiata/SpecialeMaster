import numpy as np
import itertools
import time
import sys
import os
sys.path.append(os.getcwd()) # added for calc server support
from multiprocess import Pool, cpu_count

from Thesis.Heston import AndersenLake as al, HestonModel as hm, Sobol
from Thesis.misc import VanillaOptions as vo

def calc_imp_vol(input_array : np.array, option_list : np.array) -> (np.array, np.array):
    some_model = hm.HestonClass(input_array[0], input_array[1], input_array[2], input_array[3], input_array[4], input_array[5], input_array[6])
    output_lenght = np.shape(option_list)[0]
    output_price = np.empty(output_lenght, dtype=np.float64)
    output_imp_vol = np.empty(output_lenght, dtype=np.float64)
    
    for i in range(output_lenght):
        try:
            output_price[i] = al.Andersen_Lake(some_model, option_list[i])
            output_imp_vol[i] = some_model.impVol(output_price[i], option_list[i])
        except: #overflow in char function, set impvol to 0
            output_price[i] = 0
            output_imp_vol[i] = 0

    return output_price, output_imp_vol

def imp_vol_generator(input_array : np.ndarray, option_list : np.array) -> (np.ndarray, np.ndarray):
    output_price_matrix = np.empty([np.shape(input_array)[0], np.shape(option_list)[0]])
    output_imp_vol_matrix = np.empty([np.shape(input_array)[0], np.shape(option_list)[0]])
    i = 0
    for someInput in input_array:
        output_price_matrix[i, :], output_imp_vol_matrix[i, :] = calc_imp_vol(someInput, option_list)
        i += 1
    
    return output_price_matrix, output_imp_vol_matrix

def model_input_generator() -> np.ndarray:
    # Forward
    forward = np.linspace(start = 50, stop = 150, num = 6)

    # vol
    vol = np.linspace(start = 0.01, stop = 0.2, num = 6)

    # kappa
    kappa = np.linspace(start = 0.1, stop = 2, num = 6)

    # theta
    theta = np.linspace(start = 0.01, stop = 0.2, num = 6)

    # epsilon
    epsilon = np.linspace(start = 0.1, stop = 2, num = 6)

    # rho
    rho = np.linspace(start = -0.99, stop = 0.99, num = 6)

    # rate
    rate = np.linspace(start = 0, stop = 0.2, num = 6)

    return np.array(list(itertools.product(forward, vol, kappa, theta, epsilon, rho, rate))) # model parameter combinations

def model_input_generator_old() -> np.ndarray:
    # Forward
    forward = np.linspace(start = 50, stop = 150, num = 10)

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

def option_input_generator() -> np.ndarray:
    # Maturity
    maturity = np.linspace(start = 0.01, stop = 2, num = 5)

    # strike
    strike = np.linspace(start = 75, stop = 125, num = 5)

    return np.array(list(itertools.product(maturity, strike)))

if __name__ == "__main__":
    model_input = model_input_generator()

    option_input = option_input_generator() # different option combinations
    some_option_list = np.array([])
    for option in option_input:
        some_option_list = np.append(some_option_list, vo.EUCall(option[0], option[1]))

    # going parallel
    cpu_cores = cpu_count()
    parallel_set = np.array_split(model_input, cpu_cores, axis=0)
    parallel_list = []

    # generating list of datasets for parallel
    for i in range(cpu_cores):
        parallel_list.append((parallel_set[i], some_option_list))

    # parallel
    pool = Pool(cpu_cores)
    res = pool.starmap(imp_vol_generator, parallel_list)
    res = np.concatenate(res, axis = 1)
    price_output = res[0]
    imp_vol_output = res[1]

    # saving grid datasets
    np.savetxt("Data/hestonGridInput2_wide.csv", model_input, delimiter=",")
    np.savetxt("Data/hestonGridPrice2_wide.csv", price_output, delimiter=",")
    np.savetxt("Data/hestonGridImpVol2_wide.csv", imp_vol_output, delimiter=",")

    Sobol.generate_sobol_input(len(model_input))