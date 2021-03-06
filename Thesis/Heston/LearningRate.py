import numpy as np
import time
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras import backend as k
import tikzplotlib
import joblib
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.getcwd()) # added for calc server support

from Thesis import NeuralNetworkGenerator as nng
from Thesis.Heston import DataGeneration as dg, ModelGenerator as mg
from sklearn.model_selection import train_test_split

def NNModelNext(data_set : list, folder : str, model_name : str, n_layers : int, n_neurons : int, nn_type : str,  output_scaling : str, input_scaling : str) -> float:
    def lr_schedule(epoch, rate):
        seq = 10**np.linspace(start = -10, stop = 0, num=100)
        return seq[epoch]
        
    X_train = data_set[0] 
    X_test = data_set[1]
    Y_train = data_set[2]
    Y_test = data_set[3]

    if input_scaling == "standardize":
        norm_features = StandardScaler()
        normal_in = True
    elif input_scaling == "normalize":
        norm_features = MinMaxScaler()
        normal_in = True
    else:
        normal_in = False
    

    if output_scaling == "standardize":
        norm_labels = StandardScaler()
        normal_out = True
    elif output_scaling == "normalize":
        norm_labels = MinMaxScaler()
        normal_out = True
    else:
        normal_out = False

    if normal_in:
        X_train = norm_features.fit_transform(X_train)
        X_test = norm_features.transform(X_test)

    if normal_out:
        Y_train = norm_labels.fit_transform(Y_train)
        Y_test = norm_labels.transform(Y_test)
 
    if nn_type == "normal":
        model = nng.NN_generator(n_layers, n_neurons, np.shape(X_train)[1], np.shape(Y_train)[1])
    elif nn_type == "tanh":
        model = nng.NN_generator_tanh(n_layers, n_neurons, np.shape(X_train)[1], np.shape(Y_train)[1])
    elif nn_type == "mix":
        model = nng.NN_generator_mix(n_layers, n_neurons, np.shape(X_train)[1], np.shape(Y_train)[1])
    else:
        model = nng.NN_generator_mix_noise(n_layers, n_neurons, np.shape(X_train)[1], np.shape(Y_train)[1])

    adam = Adam()

    model.compile(
        loss = 'mean_squared_error', #mean squared error
        optimizer = adam
    )

    callbacks_list = [
        LearningRateScheduler(lr_schedule, verbose = 1)
    ]

    start_time = time.time()
    loss_history = model.fit(X_train, Y_train, epochs=100, batch_size=1024, verbose = 1, callbacks = callbacks_list, validation_split = 0.1, shuffle=True)
    stop_time = time.time()

    return loss_history

def NN_mc_model_1(data_set : list, folder : str, model_name : str, n_layers : int, n_neurons : int, nn_type : str,  output_scaling : str, input_scaling : str, include_zero : bool, special_type : str = None,) -> float:
    X_train = data_set[0] 
    X_test = data_set[1]
    Y_train = data_set[2]
    Y_test = data_set[3]

    if special_type == "mat":
        X_train, Y_train = mg.transform_mat(X_train, Y_train)
        X_test, Y_test = mg.transform_mat(X_test, Y_test)
    elif special_type == "single":
        X_train, Y_train = mg.transform_single(X_train, Y_train)
        X_test, Y_test = mg.transform_single(X_test, Y_test)

    if include_zero:
        train_index = np.all(Y_train != -1, axis = 1)
        test_index = np.all(Y_test != -1, axis = 1)
    else:
        train_index = np.all(Y_train > 0, axis = 1)
        test_index = np.all(Y_test > 0, axis = 1)
    X_train = X_train[train_index, :]
    Y_train = Y_train[train_index, :]
    X_test = X_test[test_index, :]
    Y_test = Y_test[test_index, :]
    
    data_set = [X_train, X_test, Y_train, Y_test]
    score = NNModelNext(data_set, folder, model_name, n_layers, n_neurons, nn_type,  output_scaling, input_scaling)

    return score

if __name__ == '__main__':
    train_index, test_index = mg.load_index(200000)
    
    model_input = np.loadtxt("Data/benchmark_input.csv", delimiter = ",")
    imp_vol = np.loadtxt("Data/benchmark_imp.csv", delimiter=",")

    X_train = model_input[train_index, :]
    X_test = model_input[test_index, :]
    Y_train = imp_vol[train_index, :]
    Y_test = imp_vol[test_index, :]

    data_set_1 = [X_train, X_test, Y_train, Y_test]

    x_axis = 10**np.linspace(start = -10, stop = 0, num=100)
    some_5_100_lr_run = NN_mc_model_1(data_set_1, "LR", "LR", 5, 100, "normal", "False", "standardize", False)
    some_4_1000_lr_run = NN_mc_model_1(data_set_1, "LR", "LR", 4, 1000, "normal", "False", "standardize", False)
    some_3_500_lr_run = NN_mc_model_1(data_set_1, "LR", "LR", 3, 500, "normal", "False", "standardize", False)
    some_2_500_lr_run = NN_mc_model_1(data_set_1, "LR", "LR", 2, 500, "normal", "False", "standardize", False)
    some_5_50_lr_run = NN_mc_model_1(data_set_1, "LR", "LR", 5, 50, "normal", "False", "standardize", False)
    some_5_1000_lr_run = NN_mc_model_1(data_set_1, "LR", "LR", 5, 1000, "normal", "False", "standardize", False)


    fig = plt.figure(figsize=(10, 10), dpi = 200)
    ax = fig.add_subplot(111)
    ax.plot(x_axis, some_5_100_lr_run.history["loss"], label = "5, 100")
    ax.plot(x_axis, some_4_1000_lr_run.history["loss"], label = "4, 1000")
    ax.plot(x_axis, some_3_500_lr_run.history["loss"], label = "3, 500")
    ax.plot(x_axis, some_2_500_lr_run.history["loss"], label = "2, 500")
    ax.plot(x_axis, some_5_50_lr_run.history["loss"], label = "5, 50")
    ax.plot(x_axis, some_5_1000_lr_run.history["loss"], label = "5, 1000")
    ax.set_ylabel("Loss", rotation="horizontal", labelpad=15)
    ax.set_xlabel("Learning rate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Loss w. different LR rates.")
    ax.legend(loc = "upper left")
    #ax.set_ylim(0, 0.04)
    plt.savefig("loss_lr.png")
    #tikzplotlib.save("LR_schedule2.tex")
    plt.close()