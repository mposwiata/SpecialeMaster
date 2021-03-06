import numpy as np
from multiprocessing import Pool, cpu_count, Process
import sys
import os
import itertools
sys.path.append(os.getcwd()) # added for calc server support

from Thesis.Heston import NNModelGenerator as mg

if __name__ == '__main__':
    X_train = np.loadtxt("Data/Sobol2_X_train.csv", delimiter = ",")
    X_test = np.loadtxt("Data/Sobol2_X_test.csv", delimiter = ",")
    Y_train = np.loadtxt("Data/Sobol2_Y_train.csv", delimiter = ",")
    Y_test = np.loadtxt("Data/Sobol2_Y_test.csv", delimiter = ",")

    X_train_price = np.loadtxt("Data/Sobol2_X_train_price.csv", delimiter = ",")
    X_test_price = np.loadtxt("Data/Sobol2_X_test_price.csv", delimiter = ",")
    Y_train_price = np.loadtxt("Data/Sobol2_Y_train_price.csv", delimiter = ",")
    Y_test_price = np.loadtxt("Data/Sobol2_Y_test_price.csv", delimiter = ",")

    X_train_single = np.loadtxt("Data/Sobol2_X_train_single.csv", delimiter = ",")
    X_test_single = np.loadtxt("Data/Sobol2_X_test_single.csv", delimiter = ",")
    Y_train_single = np.loadtxt("Data/Sobol2_Y_train_single.csv", delimiter = ",")
    Y_test_single = np.loadtxt("Data/Sobol2_Y_test_single.csv", delimiter = ",")

    Y_train_single = np.reshape(Y_train_single, (-1, 1))
    Y_test_single = np.reshape(Y_test_single, (-1, 1))

    data_set_single = [X_train_single, X_test_single, Y_train_single, Y_test_single]

    X_train_grid = np.loadtxt("Data/Sobol2_X_train_grid.csv", delimiter = ",")
    X_test_grid = np.loadtxt("Data/Sobol2_X_test_grid.csv", delimiter = ",")
    Y_train_grid = np.loadtxt("Data/Sobol2_Y_train_grid.csv", delimiter = ",")
    Y_test_grid = np.loadtxt("Data/Sobol2_Y_test_grid.csv", delimiter = ",")

    X_train_wide = np.loadtxt("Data/Sobol2_X_train.csv", delimiter = ",")
    X_test_wide = np.loadtxt("Data/Sobol2_X_test.csv", delimiter = ",")
    Y_train_wide = np.loadtxt("Data/Sobol2_Y_train.csv", delimiter = ",")
    Y_test_wide = np.loadtxt("Data/Sobol2_Y_test.csv", delimiter = ",")

    mc_train_index = np.loadtxt("Data/MC/train_index.csv", delimiter = ",").astype(int)
    mc_test_index = np.loadtxt("Data/MC/test_index.csv", delimiter = ",").astype(int)
    mc_input = np.loadtxt("Data/MC/HestonMC_input.csv", delimiter = ",")
    mc_output = np.loadtxt("Data/MC/HestonMC_imp_vol_10000.csv", delimiter = ",")
    X_train_mc = mc_input[mc_train_index, :]
    X_test_mc = mc_input[mc_test_index, :]
    Y_train_mc = mc_output[mc_train_index, :]
    Y_test_mc = mc_output[mc_test_index, :]

    data_set = [X_train, X_test, Y_train, Y_test]
    data_set_price = [X_train_price, X_test_price, Y_train_price, Y_test_price]
    data_set_grid = [X_train_grid, X_test_grid, Y_train_grid, Y_test_grid]
    data_set_wide = [X_train_wide, X_test_wide, Y_train_wide, Y_test_wide]
    data_set_mix = [
        np.concatenate((X_train, X_train_mc)),
        np.concatenate((X_test, X_test_mc)),
        np.concatenate((Y_train, Y_train_mc)),
        np.concatenate((Y_test, Y_test_mc))
    ]

    layers = [1, 2, 3, 4, 5]
    neurons = [50, 100, 500, 1000]

    layer_neuron_combs = np.array(list(itertools.product(layers, neurons)))

    sobol_list = list(zip(itertools.repeat(data_set), itertools.repeat("benchmark"), itertools.repeat("sobol"), \
        layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("normal"), itertools.repeat(False), itertools.repeat(False)))

    grid_list = list(zip(itertools.repeat(data_set_grid), itertools.repeat("grid_vs_sobol"), itertools.repeat("grid"), \
        layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("normal"), itertools.repeat(False), itertools.repeat(False)))

    wide_list = list(zip(itertools.repeat(data_set_wide), itertools.repeat("grid_vs_sobol"), itertools.repeat("wide"), \
        layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("normal"), itertools.repeat(False), itertools.repeat(False)))

    output_list = list(zip(itertools.repeat(data_set), itertools.repeat("output_scaling"), itertools.repeat("scaling"), \
        layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("normal"), itertools.repeat(True), itertools.repeat(False)))

    price_list = list(zip(itertools.repeat(data_set_price), itertools.repeat("price_vs_imp"), itertools.repeat("price"), \
        layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("normal"), itertools.repeat(False), itertools.repeat(False)))

    price_output_standard_list = list(zip(itertools.repeat(data_set_price), itertools.repeat("price_standard"), itertools.repeat("price"), \
        layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("normal"), itertools.repeat(True), itertools.repeat(False)))

    standard_list = list(zip(itertools.repeat(data_set), itertools.repeat("standard"), itertools.repeat("standard"), \
        layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("normal"), itertools.repeat(False), itertools.repeat(True)))

    mix_list = list(zip(itertools.repeat(data_set), itertools.repeat("activation_functions"), itertools.repeat("mix"), \
        layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("mix"), itertools.repeat(False), itertools.repeat(False)))

    tanh_list = list(zip(itertools.repeat(data_set), itertools.repeat("activation_functions"), itertools.repeat("tanh"), \
        layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("tanh"), itertools.repeat(False), itertools.repeat(False)))

    single_list = list(zip(itertools.repeat(data_set_single), itertools.repeat("single"), itertools.repeat("single"), \
        layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("normal"), itertools.repeat(False), itertools.repeat(False)))

    final_list = list(zip(itertools.repeat(data_set), itertools.repeat("final"), itertools.repeat("final"), \
        layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("mix"), itertools.repeat(True), itertools.repeat(True)))

    final_list2 = list(zip(itertools.repeat(data_set), itertools.repeat("final2"), itertools.repeat("final2"), \
        layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("mix"), itertools.repeat(False), itertools.repeat(True)))

    final_list3 = list(zip(itertools.repeat(data_set), itertools.repeat("final3"), itertools.repeat("final3"), \
        layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("mix"), itertools.repeat(True), itertools.repeat(False)))

    noise_list = list(zip(itertools.repeat(data_set), itertools.repeat("noise2"), itertools.repeat("noise"), \
        layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("noise"), itertools.repeat(False), itertools.repeat(True)))

    final_mix_data = list(zip(itertools.repeat(data_set_mix), itertools.repeat("mix_data"), itertools.repeat("mix_data"), \
        layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("mix"), itertools.repeat(False), itertools.repeat(True)))


    mac_list = final_list + final_list2 + final_list3
    server_list = price_list + standard_list + mix_list + tanh_list

    if cpu_count() == 4:
        cpu_cores = 4
    else:
        cpu_cores = int(min(cpu_count()/4, 16))

    pool = Pool(cpu_cores)
    res = pool.starmap(mg.NNModelNext, final_mix_data, chunksize=1)
    pool.close()