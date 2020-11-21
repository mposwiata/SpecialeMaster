import numpy as np
import time
from scipy.stats import norm
from multiprocessing import Pool, cpu_count, Process
from sklearn.model_selection import train_test_split
import os
import itertools
import sys
sys.path.append(os.getcwd()) # added for calc server support

from Thesis.Heston import NNModelGenerator as mg, HestonModel as hm, AndersenLake as al
from Thesis.misc import VanillaOptions as vo

def Heston_monte_carlo(some_model : hm.HestonClass, some_option : vo.VanillaOption, paths : int):
    dt = 252
    time_steps = int(some_option.tau * dt)
    forward_log = np.log(some_model.forward)
    vol = some_model.vol
    delta_t = some_option.tau / time_steps
    exp_part = np.exp(-some_model.kappa * delta_t)
    x_part = some_model.theta * (1 - exp_part)
    var_part1 = some_model.epsilon * some_model.epsilon / some_model.kappa * (exp_part - np.exp(-2 * some_model.kappa * delta_t))
    var_part2 = some_model.theta * some_model.epsilon * some_model.epsilon / (2 * some_model.kappa) * (1 - exp_part) ** 2
    corr_part = np.sqrt(1 - some_model.rho * some_model.rho)

    for i in range(time_steps):
        N_F = np.random.standard_normal(paths)

        N_v = some_model.rho * N_F + corr_part * np.random.standard_normal(paths)

        forward_log += - 0.5 * vol * delta_t + np.sqrt(vol) * np.sqrt(delta_t) * N_F

        x = vol * exp_part + x_part

        var = vol * var_part1 + var_part2

        y = np.sqrt(np.log(var / (x * x) + 1))

        vol = x * np.exp(- (y * y) / 2 + y * N_v)
    
    forward = np.exp(forward_log)

    return np.exp(-some_model.rate * some_option.tau) * (np.average(some_option(forward)))

if __name__ == '__main__':
    model_input = np.loadtxt("Data/MC/HestonMC_input.csv", delimiter=",")

    mc_imp_vol_1 = np.loadtxt("Data/MC/HestonMC_imp_vol_1.csv", delimiter=",")
    mc_imp_vol_10 = np.loadtxt("Data/MC/HestonMC_imp_vol_10.csv", delimiter=",")
    mc_imp_vol_100 = np.loadtxt("Data/MC/HestonMC_imp_vol_100.csv", delimiter=",")

    if not (os.path.exists("Data/MC/train_index.csv") and os.path.exists("Data/MC/test_index.csv")):
        index = np.arange(200000)
        train_index, test_index = train_test_split(index, test_size=0.3, random_state=42)
        np.savetxt("Data/MC/train_index.csv", train_index, delimiter=",")
        np.savetxt("Data/MC/test_index.csv", test_index, delimiter=",")
    else:
        train_index = np.loadtxt("Data/MC/train_index.csv", delimiter=",")
        test_index = np.loadtxt("Data/MC/test_index.csv", delimiter=",")

    mc_1_data = [
        model_input[train_index, :],
        model_input[test_index, :],
        mc_imp_vol_1[train_index, :],
        mc_imp_vol_1[test_index, :]
    ]

    mc_10_data = [
        model_input[train_index, :],
        model_input[test_index, :],
        mc_imp_vol_10[train_index, :],
        mc_imp_vol_10[test_index, :]
    ]

    mc_100_data = [
        model_input[train_index, :],
        model_input[test_index, :],
        mc_imp_vol_100[train_index, :],
        mc_imp_vol_100[test_index, :]
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

    mac_list = final_list + final_list2 + final_list3
    server_list = price_list + standard_list + mix_list + tanh_list

    if cpu_count() == 4:
        cpu_cores = 4
    else:
        cpu_cores = int(min(cpu_count()/4, 16))

    pool = Pool(cpu_cores)
    res = pool.starmap(mg.NNModelNext, price_output_standard_list, chunksize=1)
    pool.close()