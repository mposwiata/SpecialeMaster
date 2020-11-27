import numpy as np
from multiprocessing import Pool, cpu_count, Process
from sklearn.model_selection import train_test_split
import sys
import os
import itertools
sys.path.append(os.getcwd()) # added for calc server support

from Thesis.Heston import NNModelGenerator as mg

if __name__ == '__main__':
    if not (os.path.exists("Data/MC/train_index.csv") and os.path.exists("Data/MC/test_index.csv")):
        index = np.arange(200000)
        train_index, test_index = train_test_split(index, test_size=0.3, random_state=42)
        np.savetxt("Data/MC/train_index.csv", train_index, delimiter=",")
        np.savetxt("Data/MC/test_index.csv", test_index, delimiter=",")
    else:
        train_index = np.loadtxt("Data/MC/train_index.csv", delimiter=",").astype(int)
        test_index = np.loadtxt("Data/MC/test_index.csv", delimiter=",").astype(int)
    
    model_input_1 = np.loadtxt("Data/hestonSobolGridInput2_compare2_200000.csv", delimiter = ",")
    imp_vol_1 = np.loadtxt("Data/sobol_imp_compare200000.csv", delimiter=",")
    price_1 = np.loadtxt("Data/hestonSobolGridPrice2_compare2_200000.csv", delimiter=",")

    X_train = model_input_1[train_index, :]
    X_test = model_input_1[test_index, :]
    Y_train = imp_vol_1[train_index, :]
    Y_test = imp_vol_1[test_index, :]
    Y_train_price = price_1[train_index, :]
    Y_test_price = price_1[test_index, :]

    model_input_2 = np.loadtxt("Data/sobol_second_set_input200000.csv", delimiter = ",")
    imp_vol_2 = np.loadtxt("Data/sobol_second_set_imp_vol200000.csv", delimiter=",")

    X_train_2 = model_input_1[train_index, :]
    X_test_2 = model_input_1[test_index, :]
    Y_train_2 = imp_vol_1[train_index, :]
    Y_test_2 = imp_vol_1[test_index, :]

    data_set_1 = [X_train, X_test, Y_train, Y_test]
    data_set_2 = [X_train_2, X_test_2, Y_train_2, Y_test_2]
    data_set_price = [X_train, X_test, Y_train_price, Y_test_price]

    layers = [1, 2, 3, 4, 5]
    neurons = [50, 100, 500, 1000]

    layer_neuron_combs = np.array(list(itertools.product(layers, neurons)))
    benchmark_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("benchmark"), \
        itertools.repeat("benchmark"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat(False), itertools.repeat(False)))

    benchmark_include_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("benchmark_include"), \
        itertools.repeat("benchmark_include"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat(False), itertools.repeat(True)))

    output_scaling_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("output_scaling"), \
        itertools.repeat("output_scaling"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("standardize"), itertools.repeat(False), itertools.repeat(False)))

    output_scaling_normalize_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("output_scaling_normalize"), \
        itertools.repeat("output_scaling_normalize"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("normalize"), itertools.repeat(False), itertools.repeat(False)))

    output_scaling_mix_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("output_scaling_mix"), \
        itertools.repeat("output_scaling_mix"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat("standardize"), itertools.repeat(False), itertools.repeat(False)))

    tanh_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("tanh"), \
        itertools.repeat("tanh"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("tanh"), itertools.repeat("False"), itertools.repeat(False), itertools.repeat(False)))

    mix_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("mix"), \
        itertools.repeat("mix"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat("False"), itertools.repeat(False), itertools.repeat(False)))

    price_list = list(zip(itertools.repeat(data_set_price), itertools.repeat("price"), \
        itertools.repeat("price"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat(False), itertools.repeat(False)))

    price_output_scaling_list = list(zip(itertools.repeat(data_set_price), itertools.repeat("output_scaling_price"), \
        itertools.repeat("output_scaling_price"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("standardize"), itertools.repeat(False), itertools.repeat(False)))

    standardize_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("standardize"), \
        itertools.repeat("standardize"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat(True), itertools.repeat(False)))

    standardize_mix_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("standardize_mix"), \
        itertools.repeat("standardize_mix"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat("False"), itertools.repeat(True), itertools.repeat(False)))

    noise_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("noise"), \
        itertools.repeat("noise"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat(True), itertools.repeat(False)))

    combined_best_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("combined"), \
        itertools.repeat("combined"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat("standardize"), itertools.repeat(True), itertools.repeat(False)))

    noise_included_standard_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("noise_included_standard"), \
        itertools.repeat("noise_included_standard"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("noise"), itertools.repeat("False"), itertools.repeat(True), itertools.repeat(True)))

    noise_included_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("noise_included"), \
        itertools.repeat("noise_included"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("noise"), itertools.repeat("False"), itertools.repeat(False), itertools.repeat(True)))

    compute10_list = benchmark_list + benchmark_include_list + output_scaling_list + tanh_list
    compute11_list = mix_list + price_list + standardize_list + noise_list
    next_list = price_output_scaling_list + combined_best_list + output_scaling_mix_list + standardize_mix_list

    next_list_2 = output_scaling_normalize_list

    if cpu_count() == 4:
        cpu_cores = 4
    else:
        cpu_cores = int(min(cpu_count()/4, 16))

    pool = Pool(cpu_cores)
    res = pool.starmap(mg.NN_mc_model_1, next_list_2, chunksize=1)
    pool.close()