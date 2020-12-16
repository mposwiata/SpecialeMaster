import numpy as np
from multiprocessing import Pool, cpu_count, Process
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
import sys
import os
import itertools
sys.path.append(os.getcwd()) # added for calc server support

from Thesis.Heston import NNModelGenerator as mg, DataGeneration as dg, ModelGenerator, ModelTesting as mt, Sobol, HestonModel as hm, AndersenLake as al
from Thesis.misc import VanillaOptions as vo

def subset(no : int, model_input : np.ndarray, imp_vol : np.ndarray) -> list:
    train_test_set = np.random.choice(np.arange(300000), size=no, replace=False)
    train_index, test_index = train_test_split(train_test_set, test_size=0.3, random_state=42)

    data_set = [
        model_input[train_index, :], model_input[test_index, :],
        imp_vol[train_index, :], imp_vol[test_index, :]
    ]

    return data_set

def sobol_self(no : int, model_input : np.ndarray, imp_vol : np.ndarray) -> list:
    train_index, test_index = train_test_split(np.arange(no), test_size=0.3, random_state=42)

    data_set = [
        model_input[train_index, :], model_input[test_index, :],
        imp_vol[train_index, :], imp_vol[test_index, :]
    ]

    return data_set

if __name__ == "__main__":
    model_input_3 = np.loadtxt("Data/sobol_final_input300000.csv", delimiter = ",")
    imp_vol_3 = np.loadtxt("Data/sobol_final_imp300000.csv", delimiter=",")
    data_set_5 = subset(5000, model_input_3, imp_vol_3)
    data_set_10 = subset(10000, model_input_3, imp_vol_3)
    data_set_25 = subset(25000, model_input_3, imp_vol_3)
    data_set_50 = subset(50000, model_input_3, imp_vol_3)
    data_set_75 = subset(75000, model_input_3, imp_vol_3)
    data_set_100 = subset(100000, model_input_3, imp_vol_3)
    data_set_125 = subset(125000, model_input_3, imp_vol_3)
    data_set_150 = subset(150000, model_input_3, imp_vol_3)
    data_set_175 = subset(175000, model_input_3, imp_vol_3)
    data_set_200 = subset(200000, model_input_3, imp_vol_3)
    data_set_225 = subset(225000, model_input_3, imp_vol_3)
    data_set_250 = subset(250000, model_input_3, imp_vol_3)
    data_set_275 = subset(275000, model_input_3, imp_vol_3)
    data_set_300 = subset(300000, model_input_3, imp_vol_3)

    sobol_set_5 = sobol_self(5000, model_input_3, imp_vol_3)
    sobol_set_10 = sobol_self(10000, model_input_3, imp_vol_3)
    sobol_set_25 = sobol_self(25000, model_input_3, imp_vol_3)
    sobol_set_50 = sobol_self(50000, model_input_3, imp_vol_3)
    sobol_set_75 = sobol_self(75000, model_input_3, imp_vol_3)
    sobol_set_100 = sobol_self(100000, model_input_3, imp_vol_3)
    sobol_set_125 = sobol_self(125000, model_input_3, imp_vol_3)
    sobol_set_150 = sobol_self(150000, model_input_3, imp_vol_3)
    sobol_set_175 = sobol_self(175000, model_input_3, imp_vol_3)
    sobol_set_200 = sobol_self(200000, model_input_3, imp_vol_3)
    sobol_set_225 = sobol_self(225000, model_input_3, imp_vol_3)
    sobol_set_250 = sobol_self(250000, model_input_3, imp_vol_3)
    sobol_set_275 = sobol_self(275000, model_input_3, imp_vol_3)
    sobol_set_300 = sobol_self(300000, model_input_3, imp_vol_3)

    model_list = [
        [data_set_5, "loss_history", "data_set_5", 5, 500, "normal", "False", "standardize", False],
        [data_set_10, "loss_history", "data_set_10", 5, 500, "normal", "False", "standardize", False],
        [data_set_25, "loss_history", "data_set_25", 5, 500, "normal", "False", "standardize", False],
        [data_set_50, "loss_history", "data_set_50", 5, 500, "normal", "False", "standardize", False],
        [data_set_75, "loss_history", "data_set_75", 5, 500, "normal", "False", "standardize", False],
        [data_set_100, "loss_history", "data_set_100", 5, 500, "normal", "False", "standardize", False],
        [data_set_125, "loss_history", "data_set_125", 5, 500, "normal", "False", "standardize", False],
        [data_set_150, "loss_history", "data_set_150", 5, 500, "normal", "False", "standardize", False],
        [data_set_175, "loss_history", "data_set_175", 5, 500, "normal", "False", "standardize", False],
        [data_set_200, "loss_history", "data_set_200", 5, 500, "normal", "False", "standardize", False],
        [data_set_225, "loss_history", "data_set_225", 5, 500, "normal", "False", "standardize", False],
        [data_set_250, "loss_history", "data_set_250", 5, 500, "normal", "False", "standardize", False],
        [data_set_275, "loss_history", "data_set_275", 5, 500, "normal", "False", "standardize", False],
        [data_set_300, "loss_history", "data_set_300", 5, 500, "normal", "False", "standardize", False],
        [sobol_set_5, "loss_history", "sobol_set_5", 5, 500, "normal", "False", "standardize", False],
        [sobol_set_10, "loss_history", "sobol_set_10", 5, 500, "normal", "False", "standardize", False],
        [sobol_set_25, "loss_history", "sobol_set_25", 5, 500, "normal", "False", "standardize", False],
        [sobol_set_50, "loss_history", "sobol_set_50", 5, 500, "normal", "False", "standardize", False],
        [sobol_set_75, "loss_history", "sobol_set_75", 5, 500, "normal", "False", "standardize", False],
        [sobol_set_100, "loss_history", "sobol_set_100", 5, 500, "normal", "False", "standardize", False],
        [sobol_set_125, "loss_history", "sobol_set_125", 5, 500, "normal", "False", "standardize", False],
        [sobol_set_150, "loss_history", "sobol_set_150", 5, 500, "normal", "False", "standardize", False],
        [sobol_set_175, "loss_history", "sobol_set_175", 5, 500, "normal", "False", "standardize", False],
        [sobol_set_200, "loss_history", "sobol_set_200", 5, 500, "normal", "False", "standardize", False],
        [sobol_set_225, "loss_history", "sobol_set_225", 5, 500, "normal", "False", "standardize", False],
        [sobol_set_250, "loss_history", "sobol_set_250", 5, 500, "normal", "False", "standardize", False],
        [sobol_set_275, "loss_history", "sobol_set_275", 5, 500, "normal", "False", "standardize", False],
        [sobol_set_300, "loss_history", "sobol_set_300", 5, 500, "normal", "False", "standardize", False]
    ]

    if cpu_count() == 4:
        cpu_cores = 4
    else:
        cpu_cores = int(min(cpu_count()/4, 16))

    pool = Pool(cpu_cores)
    res = pool.starmap(mg.NN_mc_model_1, model_list, chunksize=1)
    pool.close()

    """
    ### Cross checking models
    model_list = [
        "Models5/loss_history/benchmark_5_500.h5",
        "Models5/loss_history/benchmark2_5_500.h5",
        "Models5/loss_history/low_5_500.h5",
        "Models5/loss_history/low2_5_500.h5",
        "Models5/loss_history/high_5_500.h5"
    ]

    model_list_mse = mt.model_test_set(model_list, X_test_3, Y_test_3)

    model_list2_mse = mt.model_test_set(model_list, X_test, Y_test)

    model_list3_mse = mt.model_test_set(model_list, X_test_1, Y_test_1)

    random_mse = mt.model_test_set(model_list, X_test_random, Y_test_random)

    grid_sobol = [
        "Models5/grid/grid_5_500.h5",
        "Models5/sobol/sobol_5_500.h5"
    ]

    sobol_grid_random_mse = mt.model_test_set(grid_sobol, X_test_random, Y_test_random)

    imp_old = np.loadtxt("Data/benchmark_imp.csv", delimiter=",")
    imp_new = np.loadtxt("Data/sobol_final200000.csv", delimiter=",")
    """