import numpy as np
from multiprocessing import Pool, cpu_count, Process
from sklearn.model_selection import train_test_split
import sys
import os
import itertools
sys.path.append(os.getcwd()) # added for calc server support

from Thesis.Heston import NNModelGenerator as mg

def load_index(no : int) -> list:
    if not (os.path.exists("Data/MC/train_index"+str(no)+".csv") and os.path.exists("Data/MC/test_index"+str(no)+".csv")):
        index = np.arange(no)
        train_index, test_index = train_test_split(index, test_size=0.3, random_state=42)
        np.savetxt("Data/MC/train_index"+str(no)+".csv", train_index, delimiter=",")
        np.savetxt("Data/MC/test_index"+str(no)+".csv", test_index, delimiter=",")
    else:
        train_index = np.loadtxt("Data/MC/train_index"+str(no)+".csv", delimiter=",").astype(int)
        test_index = np.loadtxt("Data/MC/test_index"+str(no)+".csv", delimiter=",").astype(int)

    return train_index, test_index

if __name__ == '__main__':
    train_index, test_index = load_index(200000)
    
    model_input = np.loadtxt("Data/benchmark_input.csv", delimiter = ",")
    imp_vol = np.loadtxt("Data/benchmark_imp.csv", delimiter=",")
    price = np.loadtxt("Data/benchmark_price.csv", delimiter=",")

    X_train = model_input[train_index, :]
    X_test = model_input[test_index, :]
    Y_train = imp_vol[train_index, :]
    Y_test = imp_vol[test_index, :]
    Y_train_price = price[train_index, :]
    Y_test_price = price[test_index, :]

    data_set_1 = [X_train, X_test, Y_train, Y_test]
    data_set_price = [X_train, X_test, Y_train_price, Y_test_price]

    layers = [1, 2, 3, 4, 5]
    neurons = [50, 100, 500, 1000]

    train_index_1, test_index_1 = load_index(100000)
    model_input_1 = np.loadtxt("Data/100000_input.csv", delimiter = ",")
    imp_vol_1 = np.loadtxt("Data/100000_imp.csv", delimiter=",")
    X_train_1 = model_input_2[train_index_1, :]
    X_test_1 = model_input_2[test_index_1, :]
    Y_train_1 = imp_vol_2[train_index_1, :]
    Y_test_1 = imp_vol_2[test_index, :]

    data_set_100000 = [X_train_1, X_test_1, Y_train_1, Y_test_1]

    train_index_3, test_index_3 = load_index(300000)
    model_input_3 = np.loadtxt("Data/300000_input.csv", delimiter = ",")
    imp_vol_3 = np.loadtxt("Data/300000_imp.csv", delimiter=",")
    X_train_3 = model_input_3[train_index_1, :]
    X_test_3 = model_input_3[test_index_1, :]
    Y_train_3 = imp_vol_3[train_index_1, :]
    Y_test_3 = imp_vol_3[test_index, :]

    data_set_300000 = [X_train_3, X_test_3, Y_train_3, Y_test_3]


    layer_neuron_combs = np.array(list(itertools.product(layers, neurons)))
    benchmark_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("benchmark"), \
        itertools.repeat("benchmark"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat(False), itertools.repeat(False)))

    low_data_list = list(zip(itertools.repeat(data_set_100000), itertools.repeat("benchmark"), \
        itertools.repeat("benchmark"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat(False), itertools.repeat(False)))

    high_data_list = list(zip(itertools.repeat(data_set_300000), itertools.repeat("benchmark"), \
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

    next_list_2 = low_data_list + high_data_list

    if cpu_count() == 4:
        cpu_cores = 4
    else:
        cpu_cores = int(min(cpu_count()/4, 16))

    pool = Pool(cpu_cores)
    res = pool.starmap(mg.NN_mc_model_1, next_list_2, chunksize=1)
    pool.close()