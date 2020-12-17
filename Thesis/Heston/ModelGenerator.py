import numpy as np
from multiprocessing import Pool, cpu_count, Process
from sklearn.model_selection import train_test_split
import sys
import os
import itertools
sys.path.append(os.getcwd()) # added for calc server support

from Thesis.Heston import NNModelGenerator as mg, DataGeneration as dg

def load_index(no : int) -> list:
    if not (os.path.exists("Data/train_index_"+str(no)+".csv") and os.path.exists("Data/test_index_"+str(no)+".csv")):
        index = np.arange(no)
        train_index, test_index = train_test_split(index, test_size=0.3, random_state=42)
        np.savetxt("Data/train_index_"+str(no)+".csv", train_index, delimiter=",")
        np.savetxt("Data/test_index_"+str(no)+".csv", test_index, delimiter=",")
    else:
        train_index = np.loadtxt("Data/train_index_"+str(no)+".csv", delimiter=",").astype(int)
        test_index = np.loadtxt("Data/test_index_"+str(no)+".csv", delimiter=",").astype(int)

    return train_index, test_index

def transform_single(X_set : np.ndarray, Y_set : np.ndarray) -> (np.ndarray, np.ndarray):
    option_input = dg.option_input_generator()
    total_comb = np.shape(X_set)[0] * np.shape(Y_set)[1]
    total_cols = np.shape(X_set)[1] + 2
    total_options = np.shape(Y_set)[1]
    single_input = np.zeros((total_comb, total_cols))
    single_output = np.zeros((total_comb, 1))
    for i in range(np.shape(X_set)[0]):
        for j in range(total_options):
            single_input[i*total_options+j, 0:np.shape(X_set)[1]] = X_set[i]
            single_input[i*total_options+j, (np.shape(X_set)[1]) : total_cols] = option_input[j]
            single_output[i*total_options+j] = Y_set[i, j]
    
    return single_input, single_output

def transform_mat(X_set : np.ndarray, Y_set : np.ndarray) -> (np.ndarray, np.ndarray):
    option_input = dg.option_input_generator()
    total_comb = np.shape(X_set)[0] * 5
    total_cols = np.shape(X_set)[1] + 1
    total_options = 5
    mat_input = np.zeros((total_comb, total_cols))
    mat_output = np.zeros((total_comb, 5))
    for i in range(np.shape(X_set)[0]):
        for j in range(5):
            mat_input[i*total_options+j, 0:np.shape(X_set)[1]] = X_set[i]
            mat_input[i*total_options+j, (np.shape(X_set)[1]) : total_cols] = option_input[j*5, 0]
            mat_output[i*total_options+j,:] = Y_set[i, j*5 : j*5+5]
    
    return mat_input, mat_output

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
    X_train_single, Y_train_single = transform_single(X_train, Y_train)
    X_test_single, Y_test_single = transform_single(X_test, Y_test)
    X_train_mat, Y_train_mat = transform_mat(X_train, Y_train)
    X_test_mat, Y_test_mat = transform_mat(X_test, Y_test)

    data_set_1 = [X_train, X_test, Y_train, Y_test]
    data_set_price = [X_train, X_test, Y_train_price, Y_test_price]
    data_set_single = [X_train_single, X_test_single, Y_train_single, Y_test_single]
    data_set_mat = [X_train_mat, X_test_mat, Y_train_mat, Y_test_mat]

    ### Random
    random_input = np.loadtxt("Data/random_input.csv", delimiter = ",")
    #imp_vol = np.loadtxt("Data/benchmark_imp.csv", delimiter=",")
    random_imp_vol = np.loadtxt("Data/random_imp.csv", delimiter=",")

    X_train_random = random_input[train_index, :]
    X_test_random = random_input[test_index, :]
    Y_train_random = random_imp_vol[train_index, :]
    Y_test_random = random_imp_vol[test_index, :]

    data_set_random = [X_train_random, X_test_random, Y_train_random, Y_test_random]

    layers = [1, 2, 3, 4, 5]
    neurons = [50, 100, 500, 1000]

    train_index_1, test_index_1 = load_index(100000)
    model_input_1 = np.loadtxt("Data/sobol_final_input100000.csv", delimiter = ",")
    imp_vol_1 = np.loadtxt("Data/sobol_final_imp100000.csv", delimiter=",")
    X_train_1 = model_input_1[train_index_1, :]
    X_test_1 = model_input_1[test_index_1, :]
    Y_train_1 = imp_vol_1[train_index_1, :]
    Y_test_1 = imp_vol_1[test_index_1, :]

    data_set_100000 = [X_train_1, X_test_1, Y_train_1, Y_test_1]

    train_index_3, test_index_3 = load_index(300000)
    model_input_3 = np.loadtxt("Data/sobol_final_input300000.csv", delimiter = ",")
    imp_vol_3 = np.loadtxt("Data/sobol_final_imp300000.csv", delimiter=",")
    X_train_3 = model_input_3[train_index_3, :]
    X_test_3 = model_input_3[test_index_3, :]
    Y_train_3 = imp_vol_3[train_index_3, :]
    Y_test_3 = imp_vol_3[test_index_3, :]

    data_set_300000 = [X_train_3, X_test_3, Y_train_3, Y_test_3]

    train_grid_compare_index, test_grid_compare_index  = load_index(279936)
    grid_compare_sobol_input = np.loadtxt("Data/sobol_final_input279936.csv", delimiter = ",")
    grid_compare_sobol_imp = np.loadtxt("Data/sobol_final_imp279936.csv", delimiter = ",")
    grid_compare_input = np.loadtxt("Data/grid_input.csv", delimiter = ",")
    grid_compare_imp = np.loadtxt("Data/grid_imp.csv", delimiter = ",")
    X_train_grid_sobol = grid_compare_sobol_input[train_grid_compare_index, :]
    X_test_grid_sobol = grid_compare_sobol_input[test_grid_compare_index, :]
    Y_train_grid_sobol = grid_compare_sobol_imp[train_grid_compare_index, :]
    Y_test_grid_sobol = grid_compare_sobol_imp[test_grid_compare_index, :]

    X_train_grid = grid_compare_input[train_grid_compare_index, :]
    X_test_grid = grid_compare_input[test_grid_compare_index, :]
    Y_train_grid = grid_compare_imp[train_grid_compare_index, :]
    Y_test_grid = grid_compare_imp[test_grid_compare_index, :]

    data_set_grid_sobol = [X_train_grid_sobol, X_test_grid_sobol, Y_train_grid_sobol, Y_test_grid_sobol]
    data_set_grid = [X_train_grid, X_test_grid, Y_train_grid, Y_test_grid]

    layer_neuron_combs = np.array(list(itertools.product(layers, neurons)))
    benchmark_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("benchmark"), \
        itertools.repeat("benchmark"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("normalize"), itertools.repeat(False)))

    non_input_scaling_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("non_input_scaling"), \
        itertools.repeat("non_input_scaling"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("False"), itertools.repeat(False)))

    low_data_list = list(zip(itertools.repeat(data_set_100000), itertools.repeat("low_data"), \
        itertools.repeat("low_data"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("normalize"), itertools.repeat(False)))

    high_data_list = list(zip(itertools.repeat(data_set_300000), itertools.repeat("high_data"), \
        itertools.repeat("high_data"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("normalize"), itertools.repeat(False)))

    random_data_list = list(zip(itertools.repeat(data_set_random), itertools.repeat("random_data2"), \
        itertools.repeat("random_data2"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("normalize"), itertools.repeat(False)))

    grid_sobol_list = list(zip(itertools.repeat(data_set_grid_sobol), itertools.repeat("grid_sobol"), \
        itertools.repeat("grid_sobol"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("normalize"), itertools.repeat(False)))

    grid_list = list(zip(itertools.repeat(data_set_grid), itertools.repeat("grid"), \
        itertools.repeat("grid"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("normalize"), itertools.repeat(False)))

    benchmark_include_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("benchmark_include"), \
        itertools.repeat("benchmark_include"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("normalize"), itertools.repeat(True)))

    output_scaling_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("output_scaling"), \
        itertools.repeat("output_scaling"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("standardize"), itertools.repeat("normalize"), itertools.repeat(False)))

    output_scaling_normalize_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("output_scaling_normalize"), \
        itertools.repeat("output_scaling_normalize"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("normalize"), itertools.repeat("normalize"), itertools.repeat(False)))

    output_scaling_mix_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("output_scaling_mix"), \
        itertools.repeat("output_scaling_mix"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat("standardize"), itertools.repeat("normalize"), itertools.repeat(False)))

    tanh_list = tanh_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("tanh"), \
        itertools.repeat("tanh"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("tanh"), itertools.repeat("False"), itertools.repeat("normalize"), itertools.repeat(False)))

    standardize_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("standardize"), \
        itertools.repeat("standardize"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False)))

    standardize_non_early_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("standardize_non_early"), \
        itertools.repeat("standardize_non_early"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False)))

    standardize_single_list = list(zip(itertools.repeat(data_set_single), itertools.repeat("standardize_single"), \
        itertools.repeat("standardize_single"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False)))

    standardize_mat_list = list(zip(itertools.repeat(data_set_mat), itertools.repeat("standardize_mat"), \
        itertools.repeat("standardize_mat"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False)))

    tanh_standardize_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("tanh_standardize"), \
        itertools.repeat("tanh_standardize"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("tanh"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False)))

    mix_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("mix"), \
        itertools.repeat("mix"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat("False"), itertools.repeat("normalize"), itertools.repeat(False)))

    mix_standardize_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("mix_standardize"), \
        itertools.repeat("mix_standardize"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False)))

    price_list = list(zip(itertools.repeat(data_set_price), itertools.repeat("price"), \
        itertools.repeat("price"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("normalize"), itertools.repeat(False)))

    price_tanh_list = list(zip(itertools.repeat(data_set_price), itertools.repeat("price_tanh"), \
        itertools.repeat("price_tanh"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("tanh"), itertools.repeat("False"), itertools.repeat("normalize"), itertools.repeat(False)))

    price_mix_list = list(zip(itertools.repeat(data_set_price), itertools.repeat("price_mix"), \
        itertools.repeat("price_mix"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat("False"), itertools.repeat("normalize"), itertools.repeat(False)))

    price_include_list = list(zip(itertools.repeat(data_set_price), itertools.repeat("price_include"), \
        itertools.repeat("price_include"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("normalize"), itertools.repeat(True)))

    price_standardize_list = list(zip(itertools.repeat(data_set_price), itertools.repeat("price_standardize"), \
        itertools.repeat("price_standardize"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False)))

    price_output_standardize_list = list(zip(itertools.repeat(data_set_price), itertools.repeat("price_output_standardize"), \
        itertools.repeat("price_output_standardize"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("standardize"), itertools.repeat("normalize"), itertools.repeat(False)))

    price_output_normalize_list = list(zip(itertools.repeat(data_set_price), itertools.repeat("price_output_normalize"), \
        itertools.repeat("price_output_normalize"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("normalize"), itertools.repeat("normalize"), itertools.repeat(False)))
           
    regul_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("regularization"), \
        itertools.repeat("regularization"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("regul"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False)))

    dropout_list = list(zip(itertools.repeat(data_set_1), itertools.repeat("dropout"), \
        itertools.repeat("dropout"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("dropout"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False)))

    same_param_list = [
        [data_set_1, "same_param", "same_param", 5, 100, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 1, 1324, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 2, 193, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 3, 139, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 4, 115, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 6, 90, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 7, 82, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 8, 76, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 9, 71, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 10, 67, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 11, 64, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 12, 61, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 13, 58, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 14, 56, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 15, 54, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 16, 52, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 17, 51, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 18, 49, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 19, 48, "normal", "False", "standardize", False],
        [data_set_1, "same_param", "same_param", 20, 47, "normal", "False", "standardize", False]
    ]

    server_list = random_data_list

    if cpu_count() == 4:
        cpu_cores = 4
    else:
        cpu_cores = int(min(cpu_count()/4, 16))

    pool = Pool(cpu_cores)
    res = pool.starmap(mg.NN_mc_model_1, server_list, chunksize=1)
    pool.close()
