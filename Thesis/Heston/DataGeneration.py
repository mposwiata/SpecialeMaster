import numpy as np
import itertools
import time
import sys
import os
sys.path.append(os.getcwd()) # added for calc server support
from multiprocessing import Pool, cpu_count

from Thesis.Heston import AndersenLake as al, HestonModel as hm, Sobol, MonteCarlo as mc
from Thesis.misc import VanillaOptions as vo

def model_input_generator() -> np.ndarray:
    # Forward
    forward = np.linspace(start = 50, stop = 150, num = 5)

    # vol
    vol = np.linspace(start = 0.01, stop = 0.2, num = 5)

    # kappa
    kappa = np.linspace(start = 0.1, stop = 2, num = 5)

    # theta
    theta = np.linspace(start = 0.01, stop = 0.2, num = 5)

    # epsilon
    epsilon = np.linspace(start = 0.1, stop = 2, num = 5)

    # rho
    rho = np.linspace(start = -0.99, stop = 0.99, num = 5)

    # rate
    rate = np.linspace(start = 0, stop = 0.2, num = 5)

    return np.array(list(itertools.product(forward, vol, kappa, theta, epsilon, rho, rate))) # model parameter combinations

def model_input_random_generator(no : int) -> np.ndarray:
    model_input = np.random.uniform(size = (no, 7))

    # Forward
    model_input[:,0] = model_input[:,0] * (150-50) + 50 # transformation from [[0,1] to [50,150]]

    # vol
    model_input[:,1] = model_input[:,1] * (0.2-0.01) + 0.01

    # kappa
    model_input[:,2] = model_input[:,2] * (2-0.1) + 0.1

    # theta
    model_input[:,3] = model_input[:,3] * (0.2-0.01) + 0.01

    # epsilon
    model_input[:,4] = model_input[:,4] * (2-0.1) + 0.1

    # rho
    model_input[:,5] = model_input[:,5] * (0.99 - (-0.99)) -0.99

    # rate
    model_input[:,6] = model_input[:,6] * 0.2

    return model_input

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
    maturity = np.linspace(start = 0.1, stop = 2, num = 5)

    # strike
    strike = np.linspace(start = 75, stop = 125, num = 5)

    return np.array(list(itertools.product(maturity, strike)))

def calc_mc_data(input_array : np.array, paths : int) -> (np.array, np.array):
    option_array = option_input_generator()
    some_option_list = np.array([])
    for option in option_array:
        some_option_list = np.append(some_option_list, vo.EUCall(option[0], option[1]))

    some_model = hm.HestonClass(input_array[0], input_array[1], input_array[2], input_array[3], input_array[4], input_array[5], input_array[6])
    output_lenght = np.shape(some_option_list)[0]
    output_price = np.empty(output_lenght, dtype=np.float64)
    output_imp_vol = np.empty(output_lenght, dtype=np.float64)
    for i in range(output_lenght):
        output_price[i] = mc.Heston_monte_carlo(some_model, some_option_list[i], paths)
        output_imp_vol[i] = some_model.impVol(output_price[i], some_option_list[i])

    return output_price, output_imp_vol

def monte_carlo_generator(input_array : np.ndarray, paths : int):
    option_input = option_input_generator()
    some_option_list = np.array([])
    for option in option_input:
        some_option_list = np.append(some_option_list, vo.EUCall(option[0], option[1]))
    output_price_matrix = np.empty([np.shape(input_array)[0], 25])
    output_imp_vol_matrix = np.empty([np.shape(input_array)[0], 25])
    i = 0
    for someInput in input_array:
        output_price_matrix[i, :], output_imp_vol_matrix[i, :] = calc_mc_data(someInput, paths)
        i += 1
    
    return output_price_matrix, output_imp_vol_matrix

def calc_imp_vol(input_array : np.array, option_list : list) -> (np.array, np.array):
    some_model = hm.HestonClass(input_array[0], input_array[1], input_array[2], input_array[3], input_array[4], input_array[5], input_array[6])
    output_lenght = np.shape(option_list)[0]
    output_price = np.zeros(output_lenght, dtype=np.float64)
    output_imp_vol = np.zeros(output_lenght, dtype=np.float64)
    
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

if __name__ == "__main__":
    #model_input = model_input_generator()
    model_input = model_input_random_generator(200000)

    option_input = option_input_generator() # different option combinations
    some_option_list = np.array([])
    for option in option_input:
        some_option_list = np.append(some_option_list, vo.EUCall(option[0], option[1]))

    # going parallel
    if cpu_count() == 4:
        cpu_cores = 4
    else:
        cpu_cores = int(min(cpu_count()/4, 16))
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
    np.savetxt("Data/random_input_200000.csv", model_input, delimiter=",")
    np.savetxt("Data/random_price_200000.csv", price_output, delimiter=",")
    np.savetxt("Data/random_imp_200000.csv", imp_vol_output, delimiter=",")
    """
    ### Grid sequence
    model_input = model_input_generator()

    option_input = option_input_generator() # different option combinations
    some_option_list = np.array([])
    for option in option_input:
        some_option_list = np.append(some_option_list, vo.EUCall(option[0], option[1]))

    # going parallel
    if cpu_count() == 4:
        cpu_cores = 4
    else:
        cpu_cores = int(min(cpu_count()/4, 16))
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
    np.savetxt("Data/grid_input.csv", model_input, delimiter=",")
    np.savetxt("Data/grid_price.csv", price_output, delimiter=",")
    np.savetxt("Data/grid_imp.csv", imp_vol_output, delimiter=",")
    """
