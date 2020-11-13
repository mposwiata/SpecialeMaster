import numpy as np
import sobol_seq as sobol
import itertools
import time
import sys
import os
sys.path.append(os.getcwd()) # added for calc server support
from multiprocess import Pool, cpu_count

from Thesis.Heston import AndersenLake as al, HestonModel as hm, DataGeneration as dg
from Thesis.misc import VanillaOptions as vo

def generate_sobol_input(no_sobol : int):
    model_input = sobol.i4_sobol_generate(7, no_sobol) # model input of sobol sequences

    # Forward
    model_input[:,0] = model_input[:,0] * (150-50) + 50 # transformation from [0,1] to [75,125]

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

    option_input = dg.option_input_generator() # different option combinations
    some_option_list = np.array([])
    for option in option_input:
        some_option_list = np.append(some_option_list, vo.EUCall(option[0], option[1]))

    # generating data for neural net with model as input and grid as output
    gridInput = model_input

    # going parallel
    cpu_cores = cpu_count()
    parallel_set = np.array_split(model_input, cpu_cores, axis=0)
    parallel_list = []

    # generating list of datasets for parallel
    for i in range(cpu_cores):
        parallel_list.append((parallel_set[i], some_option_list))

    # parallel
    pool = Pool(cpu_cores)
    res = pool.starmap(dg.imp_vol_generator, parallel_list)
    res = np.concatenate(res, axis = 1)
    price_output = res[0]
    imp_vol_output = res[1]

    # saving grid datasets
    np.savetxt("Data/hestonSobolGridInput2_compare_"+str(no_sobol)+".csv", gridInput, delimiter=",")
    np.savetxt("Data/hestonSobolGridPrice2_compare_"+str(no_sobol)+".csv", price_output, delimiter=",")
    np.savetxt("Data/hestonSobolGridImpVol2_compare_"+str(no_sobol)+".csv", imp_vol_output, delimiter=",")

    return 0

if __name__ == "__main__":
    print("nothing")
    #generate_sobol_input(100000)
    #generate_sobol_input(200000)
    #generate_sobol_input(312500)
    
    #generate_sobol_input(279936) # matching number of inputs for the wide model
