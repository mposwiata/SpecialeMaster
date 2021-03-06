import numpy as np
import time
from scipy.stats import norm
from multiprocessing import Pool, cpu_count, Process
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import os
import itertools
import sys
sys.path.append(os.getcwd()) # added for calc server support

from Thesis.Heston import NNModelGenerator as mg, HestonModel as hm, AndersenLake as al, ModelGenerator
from Thesis.misc import VanillaOptions as vo

def Heston_monte_carlo(some_model : hm.HestonClass, some_option : vo.VanillaOption, paths : int, use_parity : bool = False):
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

    if (np.average(forward) < some_option.strike and use_parity):
        some_put = vo.EUPut(some_option.tau, some_option.strike)
        put_price = np.exp(-some_model.rate * some_option.tau) * (np.average(some_put(forward)))
        print(put_price)
        call_price = put_price - np.exp(-some_model.rate * some_option.tau) * (some_option.strike - some_model.forward)
    else:
        call_price = np.exp(-some_model.rate * some_option.tau) * (np.average(some_option(forward)))

    return call_price

if __name__ == '__main__':
    test_model = hm.HestonClass(50, 0.01, 0.1, 0.01, 2, 0.8, 0.05)
    test_option = vo.EUCall(0.01, 100)
    print(Heston_monte_carlo(test_model, test_option , 5000))
    print(Heston_monte_carlo(test_model, test_option , 5000, True ))
    print(al.Andersen_Lake(test_model, test_option))

    """
    model_input = np.loadtxt("Data/MC/HestonMC_input.csv", delimiter=",")

    mc_imp_vol_1 = np.loadtxt("Data/MC/Heston_mc_imp_vol_1.csv", delimiter=",")
    mc_price_1 = np.loadtxt("Data/MC/HestonMC_price_1.csv", delimiter=",")
    mc_imp_vol_10 = np.loadtxt("Data/MC/Heston_mc_imp_vol_10.csv", delimiter=",")
    mc_price_10 = np.loadtxt("Data/MC/HestonMC_price_10.csv", delimiter=",")
    mc_imp_vol_100 = np.loadtxt("Data/MC/Heston_mc_imp_vol_100.csv", delimiter=",")
    mc_price_100 = np.loadtxt("Data/MC/HestonMC_price_100.csv", delimiter=",")
    mc_imp_vol_1000 = np.loadtxt("Data/MC/Heston_mc_imp_vol_1000.csv", delimiter=",")
    mc_price_1000 = np.loadtxt("Data/MC/HestonMC_price_1000.csv", delimiter=",")
    mc_imp_vol_10000 = np.loadtxt("Data/MC/Heston_mc_imp_vol_10000.csv", delimiter=",")
    mc_price_10000 = np.loadtxt("Data/MC/HestonMC_price_10000.csv", delimiter=",")

    if not (os.path.exists("Data/train_index_200000.csv") and os.path.exists("Data/test_index_200000.csv")):
        index = np.arange(200000)
        train_index, test_index = train_test_split(index, test_size=0.3, random_state=42)
        np.savetxt("Data/train_index_200000.csv", train_index, delimiter=",")
        np.savetxt("Data/test_index_200000.csv", test_index, delimiter=",")
    else:
        train_index = np.loadtxt("Data/train_index_200000.csv", delimiter=",").astype(int)
        test_index = np.loadtxt("Data/test_index_200000.csv", delimiter=",").astype(int)

    mc_1_imp = [
        model_input[train_index, :],
        model_input[test_index, :],
        mc_imp_vol_1[train_index, :],
        mc_imp_vol_1[test_index, :]
    ]

    mc_1_price = [
        model_input[train_index, :],
        model_input[test_index, :],
        mc_price_1[train_index, :],
        mc_price_1[test_index, :]
    ]

    mc_10_imp = [
        model_input[train_index, :],
        model_input[test_index, :],
        mc_imp_vol_10[train_index, :],
        mc_imp_vol_10[test_index, :]
    ]

    mc_10_price = [
        model_input[train_index, :],
        model_input[test_index, :],
        mc_price_10[train_index, :],
        mc_price_10[test_index, :]
    ]

    mc_100_imp = [
        model_input[train_index, :],
        model_input[test_index, :],
        mc_imp_vol_100[train_index, :],
        mc_imp_vol_100[test_index, :]
    ]

    mc_100_price = [
        model_input[train_index, :],
        model_input[test_index, :],
        mc_price_100[train_index, :],
        mc_price_100[test_index, :]
    ]


    mc_1000_imp = [
        model_input[train_index, :],
        model_input[test_index, :],
        mc_imp_vol_1000[train_index, :],
        mc_imp_vol_1000[test_index, :]
    ]

    mc_1000_price = [
        model_input[train_index, :],
        model_input[test_index, :],
        mc_price_1000[train_index, :],
        mc_price_1000[test_index, :]
    ]


    mc_10000_imp = [
        model_input[train_index, :],
        model_input[test_index, :],
        mc_imp_vol_10000[train_index, :],
        mc_imp_vol_10000[test_index, :]
    ]

    mc_10000_price = [
        model_input[train_index, :],
        model_input[test_index, :],
        mc_price_10000[train_index, :],
        mc_price_10000[test_index, :]
    ]

    layers = [1, 2, 3, 4, 5]
    neurons = [50, 100, 500, 1000]

    layer_neuron_combs = np.array(list(itertools.product(layers, neurons)))

    ### Implied vols
    mc_1_set = list(zip(itertools.repeat(mc_1_imp), itertools.repeat("mc_1"), \
        itertools.repeat("mc_1"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False)))

    mc_10_set = list(zip(itertools.repeat(mc_10_imp), itertools.repeat("mc_10"), \
        itertools.repeat("mc_10"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False)))

    mc_100_set = list(zip(itertools.repeat(mc_100_imp), itertools.repeat("mc_100"), \
        itertools.repeat("mc_100"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False)))

    mc_1000_set = list(zip(itertools.repeat(mc_1000_imp), itertools.repeat("mc_1000"), \
        itertools.repeat("mc_1000"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False)))

    mc_10000_set = list(zip(itertools.repeat(mc_10000_imp), itertools.repeat("mc_10000"), \
        itertools.repeat("mc_10000"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False)))

    ### Implied vols, maturity
    mc_1_mat_set = list(zip(itertools.repeat(mc_1_imp), itertools.repeat("mc_1_mat"), \
        itertools.repeat("mc_1_mat"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False), itertools.repeat("mat")))

    mc_10_mat_set = list(zip(itertools.repeat(mc_10_imp), itertools.repeat("mc_10_mat"), \
        itertools.repeat("mc_10_mat"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False), itertools.repeat("mat")))

    mc_100_mat_set = list(zip(itertools.repeat(mc_100_imp), itertools.repeat("mc_100_mat"), \
        itertools.repeat("mc_100_mat"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False), itertools.repeat("mat")))

    mc_1000_mat_set = list(zip(itertools.repeat(mc_1000_imp), itertools.repeat("mc_1000_mat"), \
        itertools.repeat("mc_1000_mat"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False), itertools.repeat("mat")))

    mc_10000_mat_set = list(zip(itertools.repeat(mc_10000_imp), itertools.repeat("mc_10000_mat"), \
        itertools.repeat("mc_10000_mat"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False), itertools.repeat("mat")))

        ### Implied vols
    mc_1_single_set = list(zip(itertools.repeat(mc_1_imp), itertools.repeat("mc_1_single"), \
        itertools.repeat("mc_1_single"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False), itertools.repeat("single")))

    mc_10_single_set = list(zip(itertools.repeat(mc_10_imp), itertools.repeat("mc_10_single"), \
        itertools.repeat("mc_10_single"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False), itertools.repeat("single")))

    mc_100_single_set = list(zip(itertools.repeat(mc_100_imp), itertools.repeat("mc_100_single"), \
        itertools.repeat("mc_100_single"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False), itertools.repeat("single")))

    mc_1000_single_set = list(zip(itertools.repeat(mc_1000_imp), itertools.repeat("mc_1000_single"), \
        itertools.repeat("mc_1000_single"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False), itertools.repeat("single")))

    mc_10000_single_set = list(zip(itertools.repeat(mc_10000_imp), itertools.repeat("mc_10000_single"), \
        itertools.repeat("mc_10000_single"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("normal"), itertools.repeat("False"), itertools.repeat("standardize"), itertools.repeat(False), itertools.repeat("single")))

    imp_list = mc_1_set + mc_10_set + mc_100_set + mc_1000_set + mc_10000_set
    imp_mat_list = mc_1_mat_set + mc_10_mat_set + mc_100_mat_set + mc_1000_mat_set + mc_10000_mat_set
    imp_single_list = mc_1_single_set + mc_10_single_set + mc_100_single_set + mc_1000_single_set + mc_10000_single_set

    ### Prices
    mc_1_price_set = list(zip(itertools.repeat(mc_1_price), itertools.repeat("mc_1_price"), \
        itertools.repeat("mc_1_price"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat("normalize"), itertools.repeat("standardize"), itertools.repeat(True)))

    mc_10_price_set = list(zip(itertools.repeat(mc_10_price), itertools.repeat("mc_10_price"), \
        itertools.repeat("mc_10_price"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat("normalize"), itertools.repeat("standardize"), itertools.repeat(True)))

    mc_100_price_set = list(zip(itertools.repeat(mc_100_price), itertools.repeat("mc_100_price"), \
        itertools.repeat("mc_100_price"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat("normalize"), itertools.repeat("standardize"), itertools.repeat(True)))

    mc_1000_price_set = list(zip(itertools.repeat(mc_1000_price), itertools.repeat("mc_1000_price"), \
        itertools.repeat("mc_1000_price"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat("normalize"), itertools.repeat("standardize"), itertools.repeat(True)))

    mc_10000_price_set = list(zip(itertools.repeat(mc_10000_price), itertools.repeat("mc_10000_price"), \
        itertools.repeat("mc_10000_price"), layer_neuron_combs[:, 0], layer_neuron_combs[:, 1], \
        itertools.repeat("mix"), itertools.repeat("normalize"), itertools.repeat("standardize"), itertools.repeat(True)))

    price_list = mc_1_price_set + mc_10_price_set + mc_100_price_set + mc_1000_price_set + mc_10000_price_set

    server_list = imp_mat_list + imp_single_list

    if cpu_count() == 4:
        cpu_cores = 4
    else:
        cpu_cores = int(min(cpu_count()/4, 16))

    pool = Pool(cpu_cores)
    res = pool.starmap(mg.NN_mc_model_1, server_list, chunksize=1)
    pool.close()
    """