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

if __name__ == "__main__":
    train_index, test_index = ModelGenerator.load_index(200000)
    
    model_input = np.loadtxt("Data/benchmark_input.csv", delimiter = ",")
    imp_vol = np.loadtxt("Data/benchmark_imp.csv", delimiter=",")
    price = np.loadtxt("Data/benchmark_price.csv", delimiter=",")
    price2 = np.loadtxt("Data/sobol_second_set_price200000.csv", delimiter=",")

    X_train = model_input[train_index, :]
    X_test = model_input[test_index, :]
    Y_train = imp_vol[train_index, :]
    Y_test = imp_vol[test_index, :]

    data_set_1 = [X_train, X_test, Y_train, Y_test]

    train_index_1, test_index_1 = ModelGenerator.load_index(100000)
    model_input_1 = np.loadtxt("Data/100000_input.csv", delimiter = ",")
    imp_vol_1 = np.loadtxt("Data/100000_imp.csv", delimiter=",")
    X_train_1 = model_input_1[train_index_1, :]
    X_test_1 = model_input_1[test_index_1, :]
    Y_train_1 = imp_vol_1[train_index_1, :]
    Y_test_1 = imp_vol_1[test_index_1, :]

    data_set_100000 = [X_train_1, X_test_1, Y_train_1, Y_test_1]

    train_index_3, test_index_3 = ModelGenerator.load_index(300000)
    model_input_3 = np.loadtxt("Data/300000_input.csv", delimiter = ",")
    imp_vol_3 = np.loadtxt("Data/300000_imp.csv", delimiter=",")
    X_train_3 = model_input_3[train_index_3, :]
    X_test_3 = model_input_3[test_index_3, :]
    Y_train_3 = imp_vol_3[train_index_3, :]
    Y_test_3 = imp_vol_3[test_index_3, :]

    data_set_300000 = [X_train_3, X_test_3, Y_train_3, Y_test_3]

    ### Pulling 100k from the 300k
    len_100k = len(train_index_1) + len(test_index_1)
    train_test_100k = np.random.choice(np.arange(300000), size=len_100k, replace=False)
    train_100k2, test_100k2 = train_test_split(train_test_100k, test_size=0.3, random_state=42)

    data_set_300_100 = [
        model_input_3[train_100k2, :], model_input_3[test_100k2, :],
        imp_vol_3[train_100k2, :], imp_vol_3[test_100k2, :]
    ]

    ### Pulling 200k from the 300k
    len_200k = len(train_index) + len(test_index)
    train_test_200k = np.random.choice(np.arange(300000), size=len_200k, replace=False)
    train_200k2, test_200k2 = train_test_split(train_test_200k, test_size=0.3, random_state=42)

    data_set_300_200 = [
        model_input_3[train_200k2, :], model_input_3[test_200k2, :],
        imp_vol_3[train_200k2, :], imp_vol_3[test_200k2, :]
    ]

    benchmark_loss, benchmark_score = mg.NN_mc_model_1(data_set_1, "loss_history", "benchmark", 5, 500, \
        "normal", "False", "standardize", False, None, True)

    low_loss, low_score = mg.NN_mc_model_1(data_set_100000, "loss_history", "low", 5, 500, \
        "normal", "False", "standardize", False, None, True)

    low2_loss, low2_score = mg.NN_mc_model_1(data_set_300_100, "loss_history", "low2", 5, 500, \
        "normal", "False", "standardize", False, None, True)

    high_loss, high_score = mg.NN_mc_model_1(data_set_300000, "loss_history", "high", 5, 500, \
        "normal", "False", "standardize", False, None, True)

    benchmark2_loss, benchmark2_score = mg.NN_mc_model_1(data_set_300_200, "loss_history", "benchmark2", 5, 500, \
        "normal", "False", "standardize", False, None, True)

    val_loss = np.array((low2_loss.history["val_loss"][-1], benchmark2_loss.history["val_loss"][-1], high_loss.history["val_loss"][-1]))
    loss = np.array((low2_loss.history["loss"][-1], benchmark2_loss.history["loss"][-1], high_loss.history["loss"][-1]))
    obs = np.array((100, 200, 300))

    fig = plt.figure(figsize=(10, 10), dpi = 200)
    ax = plt.subplot(111)
    ax.plot(benchmark2_loss.history["loss"], label = "200k subset loss")
    ax.plot(benchmark2_loss.history["val_loss"], label = "200k subset val loss")
    ax.plot(benchmark_loss.history["loss"], label = "200k Sobol loss")
    ax.plot(benchmark_loss.history["val_loss"], label = "200k Sobol val loss")
    ax.plot(high_loss.history["loss"], label = "300k Sobol loss")
    ax.plot(high_loss.history["val_loss"], label = "300k Sobol val loss")
    ax.plot(low2_loss.history["loss"], label = "100k subset loss")
    ax.plot(low2_loss.history["val_loss"], label = "100k subset val loss")
    ax.plot(low_loss.history["loss"], label = "100k Sobol loss")
    ax.plot(low_loss.history["val_loss"], label = "100k Sobol val loss")
        
    fig.suptitle("Val vs training loss",fontsize=25)
    ax.tick_params(axis="both",labelsize=15)
    ax.yaxis.offsetText.set_fontsize(15)
    ax.set_xlabel("Epoch", fontsize=20)
    ax.set_ylabel("Loss", fontsize=20)
    ax.legend(loc="lower left", fontsize=15)
    ax.set_ylim(0, 1e-4)
    plt.savefig("val_training_loss.png")
    plt.close()

    fig = plt.figure(figsize=(10, 10), dpi = 200)
    ax = plt.subplot(111)
    ax.plot(obs, val_loss, label = "val loss")
    ax.plot(obs, loss, label = "loss")
        
    fig.suptitle("Val vs training loss",fontsize=25)
    ax.tick_params(axis="both",labelsize=15)
    ax.yaxis.offsetText.set_fontsize(15)
    ax.set_xlabel("Epoch", fontsize=20)
    ax.set_ylabel("Loss", fontsize=20)
    ax.legend(loc="lower left", fontsize=15)
    ax.set_ylim(0, 1e-4)
    plt.savefig("val_training_loss_numbers.png")
    plt.close()

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
