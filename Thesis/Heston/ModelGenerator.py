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

    X_train = model_input_1[train_index, :]
    X_test = model_input_1[test_index, :]
    Y_train = imp_vol_1[train_index, :]
    Y_test = imp_vol_1[test_index, :]

    model_input_2 = np.loadtxt("Data/sobol_second_set_input200000.csv", delimiter = ",")
    imp_vol_2 = np.loadtxt("Data/sobol_second_set_imp_vol200000.csv", delimiter=",")

    X_train_2 = model_input_1[train_index, :]
    X_test_2 = model_input_1[test_index, :]
    Y_train_2 = imp_vol_1[train_index, :]
    Y_test_2 = imp_vol_1[test_index, :]

    data_set_1 = [X_train, X_test, Y_train, Y_test]
    data_set_2 = [X_train_2, X_test_2, Y_train_2, Y_test_2]

    layers = [1, 2, 3, 4, 5]
    neurons = [50, 100, 500, 1000]

    layer_neuron_combs = np.array(list(itertools.product(layers, neurons)))

    model_list_1 = list(zip(itertools.repeat(data_set_1), itertools.repeat("new_data_include_zero"), \
        itertools.repeat("include_zero"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat(False), itertools.repeat(True), itertools.repeat(True)))

    model_list_2 = list(zip(itertools.repeat(data_set_1), itertools.repeat("new_data"), \
        itertools.repeat("new_data"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat(False), itertools.repeat(True), itertools.repeat(False)))

    model_list_3 = list(zip(itertools.repeat(data_set_2), itertools.repeat("new_option_indluce"), \
        itertools.repeat("include_zero"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat(False), itertools.repeat(True), itertools.repeat(True)))

    model_list_4 = list(zip(itertools.repeat(data_set_2), itertools.repeat("new_option"), \
        itertools.repeat("new_option"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat(False), itertools.repeat(True), itertools.repeat(False)))


    server_list = model_list_1 + model_list_2 + model_list_3 + model_list_4

    if cpu_count() == 4:
        cpu_cores = 4
    else:
        cpu_cores = int(min(cpu_count()/4, 16))

    pool = Pool(cpu_cores)
    res = pool.starmap(mg.NN_mc_model_1, server_list, chunksize=1)
    pool.close()