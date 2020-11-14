import numpy as np
from multiprocessing import Pool, cpu_count, Process
import sys
import os
import itertools
sys.path.append(os.getcwd()) # added for calc server support

from Thesis.Heston import NNModelGenerator as mg


X_train = np.loadtxt("Data/Sobol2_X_train.csv", delimiter = ",")
X_test = np.loadtxt("Data/Sobol2_X_test.csv", delimiter = ",")
Y_train = np.loadtxt("Data/Sobol2_Y_train.csv", delimiter = ",")
Y_test = np.loadtxt("Data/Sobol2_Y_test.csv", delimiter = ",")

X_train_price = np.loadtxt("Data/Sobol2_X_train_price.csv", delimiter = ",")
X_test_price = np.loadtxt("Data/Sobol2_X_test_price.csv", delimiter = ",")
Y_train_price = np.loadtxt("Data/Sobol2_Y_train_price.csv", delimiter = ",")
Y_test_price = np.loadtxt("Data/Sobol2_Y_test_price.csv", delimiter = ",")

"""
X_train_single = np.loadtxt("Data/Sobol2_X_train_single.csv", delimiter = ",")
X_test_single = np.loadtxt("Data/Sobol2_X_test_single.csv", delimiter = ",")
Y_train_single = np.loadtxt("Data/Sobol2_Y_train_single.csv", delimiter = ",")
Y_test_single = np.loadtxt("Data/Sobol2_Y_test_single.csv", delimiter = ",")

Y_train_single = np.reshape(Y_train_single, (-1, 1))
Y_test_single = np.reshape(Y_test_single, (-1, 1))
"""

X_train_grid = np.loadtxt("Data/Sobol2_X_train_grid.csv", delimiter = ",")
X_test_grid = np.loadtxt("Data/Sobol2_X_test_grid.csv", delimiter = ",")
Y_train_grid = np.loadtxt("Data/Sobol2_Y_train_grid.csv", delimiter = ",")
Y_test_grid = np.loadtxt("Data/Sobol2_Y_test_grid.csv", delimiter = ",")

X_train_wide = np.loadtxt("Data/Sobol2_X_train.csv", delimiter = ",")
X_test_wide = np.loadtxt("Data/Sobol2_X_test.csv", delimiter = ",")
Y_train_wide = np.loadtxt("Data/Sobol2_Y_train.csv", delimiter = ",")
Y_test_wide = np.loadtxt("Data/Sobol2_Y_test.csv", delimiter = ",")

data_set = [X_train, X_test, Y_train, Y_test]
data_set_price = [X_train_price, X_test_price, Y_train_price, Y_test_price]
data_set_grid = [X_train_grid, X_test_grid, Y_train_grid, Y_test_grid]
data_set_wide = [X_train_wide, X_test_wide, Y_train_wide, Y_test_wide]

layers = [1, 2, 3, 4, 5]
neurons = [50, 100, 500, 1000]

layer_neuron_combs = np.array(list(itertools.product(layers, neurons)))

sobol_list = list(zip(itertools.repeat(data_set), itertools.repeat("benchmark"), itertools.repeat("sobol"), \
    layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("normal"), itertools.repeat(False), itertools.repeat(False)))

grid_list = list(zip(itertools.repeat(data_set_grid), itertools.repeat("grid_vs_sobol"), itertools.repeat("grid"), \
    layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("normal"), itertools.repeat(False), itertools.repeat(False)))

wide_list = list(zip(itertools.repeat(data_set_wide), itertools.repeat("grid_vs_sobol"), itertools.repeat("sobol"), \
    layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("normal"), itertools.repeat(False), itertools.repeat(False)))

output_list = list(zip(itertools.repeat(data_set), itertools.repeat("output_scaling"), itertools.repeat("sobol"), \
    layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("normal"), itertools.repeat(True), itertools.repeat(False)))

price_list = list(zip(itertools.repeat(data_set_price), itertools.repeat("price_vs_imp"), itertools.repeat("price"), \
    layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("normal"), itertools.repeat(False), itertools.repeat(False)))

standard_list = list(zip(itertools.repeat(data_set), itertools.repeat("standard"), itertools.repeat("sobol"), \
    layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("normal"), itertools.repeat(False), itertools.repeat(True)))

mix_list = list(zip(itertools.repeat(data_set), itertools.repeat("activation_functions"), itertools.repeat("mix"), \
    layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("mix"), itertools.repeat(False), itertools.repeat(False)))

tanh_list = list(zip(itertools.repeat(data_set), itertools.repeat("activation_functions"), itertools.repeat("tanh"), \
    layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], itertools.repeat("tanh"), itertools.repeat(False), itertools.repeat(False)))

mac_list = sobol_list + grid_list + wide_list + output_list
server_list = price_list + standard_list + mix_list + tanh_list

if cpu_count() == 4:
    cpu_cores = 4
else:
    cpu_cores = int(min(cpu_count()/4, 16))

pool = Pool(cpu_cores)
res = pool.starmap(mg.NNModelNext, server_list, chunksize=1)
pool.close()