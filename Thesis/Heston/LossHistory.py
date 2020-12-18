import numpy as np
from multiprocessing import Pool, cpu_count, Process
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import sys
import os
import glob
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
        [data_set_5, "data_size2/data_set_5", "data_set_5", 3, 500, "normal", "False", "standardize", False],
        [data_set_10, "data_size2/data_set_10", "data_set_10", 3, 500, "normal", "False", "standardize", False],
        [data_set_25, "data_size2/data_set_25", "data_set_25", 3, 500, "normal", "False", "standardize", False],
        [data_set_50, "data_size2/data_set_50", "data_set_50", 3, 500, "normal", "False", "standardize", False],
        [data_set_75, "data_size2/data_set_75", "data_set_75", 3, 500, "normal", "False", "standardize", False],
        [data_set_100, "data_size2/data_set_100", "data_set_100", 3, 500, "normal", "False", "standardize", False],
        [data_set_125, "data_size2/data_set_125", "data_set_125", 3, 500, "normal", "False", "standardize", False],
        [data_set_150, "data_size2/data_set_150", "data_set_150", 3, 500, "normal", "False", "standardize", False],
        [data_set_175, "data_size2/data_set_175", "data_set_175", 3, 500, "normal", "False", "standardize", False],
        [data_set_200, "data_size2/data_set_200", "data_set_200", 3, 500, "normal", "False", "standardize", False],
        [data_set_225, "data_size2/data_set_225", "data_set_225", 3, 500, "normal", "False", "standardize", False],
        [data_set_250, "data_size2/data_set_250", "data_set_250", 3, 500, "normal", "False", "standardize", False],
        [data_set_275, "data_size2/data_set_275", "data_set_275", 3, 500, "normal", "False", "standardize", False],
        [data_set_300, "data_size2/data_set_300", "data_set_300", 3, 500, "normal", "False", "standardize", False],
        [sobol_set_5, "data_size2/sobol_set_5", "sobol_set_5", 3, 500, "normal", "False", "standardize", False],
        [sobol_set_10, "data_size2/sobol_set_10", "sobol_set_10", 3, 500, "normal", "False", "standardize", False],
        [sobol_set_25, "data_size2/sobol_set_25", "sobol_set_25", 3, 500, "normal", "False", "standardize", False],
        [sobol_set_50, "data_size2/sobol_set_50", "sobol_set_50", 3, 500, "normal", "False", "standardize", False],
        [sobol_set_75, "data_size2/sobol_set_75", "sobol_set_75", 3, 500, "normal", "False", "standardize", False],
        [sobol_set_100, "data_size2/sobol_set_100", "sobol_set_100", 3, 500, "normal", "False", "standardize", False],
        [sobol_set_125, "data_size2/sobol_set_125", "sobol_set_125", 3, 500, "normal", "False", "standardize", False],
        [sobol_set_150, "data_size2/sobol_set_150", "sobol_set_150", 3, 500, "normal", "False", "standardize", False],
        [sobol_set_175, "data_size2/sobol_set_175", "sobol_set_175", 3, 500, "normal", "False", "standardize", False],
        [sobol_set_200, "data_size2/sobol_set_200", "sobol_set_200", 3, 500, "normal", "False", "standardize", False],
        [sobol_set_225, "data_size2/sobol_set_225", "sobol_set_225", 3, 500, "normal", "False", "standardize", False],
        [sobol_set_250, "data_size2/sobol_set_250", "sobol_set_250", 3, 500, "normal", "False", "standardize", False],
        [sobol_set_275, "data_size2/sobol_set_275", "sobol_set_275", 3, 500, "normal", "False", "standardize", False],
        [sobol_set_300, "data_size2/sobol_set_300", "sobol_set_300", 3, 500, "normal", "False", "standardize", False]
    ]

    if cpu_count() == 4:
        cpu_cores = 2
    else:
        cpu_cores = int(min(cpu_count()/4, 16))

    
    pool = Pool(cpu_cores)
    res = pool.starmap(mg.NN_mc_model_1, model_list, chunksize=1)
    pool.close()

    """
    random_input = np.loadtxt("Data/random_input_279936.csv", delimiter=",")
    random_imp = np.loadtxt("Data/random_imp_279936.csv", delimiter=",")
    test_index = np.random.choice(np.arange(len(random_imp)),size=30000, replace=False)

    # 5, 100 models
    models = glob.glob("Models5/data_size/*/*.h5")

    models_mse = mt.model_test_set(models, random_input[test_index, :], random_imp[test_index, :])

    data_set_list = []
    sobol_set_list = []

    for some_list in models_mse:
        if (some_list[0].find("data") != -1):
            data_set_list.append((some_list[0][:8], 
            int(some_list[0][9:some_list[0].rfind("_")-2]),
            some_list[1]))
        else:
            sobol_set_list.append((some_list[0][:9], 
            int(some_list[0][10:some_list[0].rfind("_")-2]),
            some_list[1]))

    data_set_list.sort(key = lambda x: x[1])
    data_set_x = [sublist[2] for sublist in data_set_list]
    data_set_y = [sublist[1] for sublist in data_set_list]
    sobol_set_list.sort(key = lambda x: x[1])
    sobol_set_x = [sublist[2] for sublist in sobol_set_list]
    sobol_set_y = [sublist[1] for sublist in sobol_set_list]

    # 3, 500 models
    models2 = glob.glob("Models5/data_size2/*/*.h5")

    models2_mse = mt.model_test_set(models2, random_input[test_index, :], random_imp[test_index, :])

    data2_set_list = []
    sobol2_set_list = []

    for some_list in models2_mse:
        if (some_list[0].find("data") != -1):
            data2_set_list.append((some_list[0][:8], 
            int(some_list[0][9:some_list[0].rfind("_")-2]),
            some_list[1]))
        else:
            sobol2_set_list.append((some_list[0][:9], 
            int(some_list[0][10:some_list[0].rfind("_")-2]),
            some_list[1]))

    data2_set_list.sort(key = lambda x: x[1])
    data2_set_x = [sublist[2] for sublist in data2_set_list]
    data2_set_y = [sublist[1] for sublist in data2_set_list]
    sobol2_set_list.sort(key = lambda x: x[1])
    sobol2_set_x = [sublist[2] for sublist in sobol2_set_list]
    sobol2_set_y = [sublist[1] for sublist in sobol2_set_list]

    fig = plt.figure(figsize=(10, 10), dpi = 200)
    ax = fig.add_subplot(111)
    ax.plot(data_set_y, data_set_x, label = "Random from Sobol, 5, 100")
    ax.plot(sobol_set_y, sobol_set_x, label = "Sobol, 5, 100")
    ax.plot(data2_set_y, data2_set_x, label = "Random from Sobol, 3, 500")
    ax.plot(sobol2_set_y, sobol2_set_x, label = "Sobol, 3, 500")
    ax.set_xlabel("Data set size", fontsize=20)
    ax.set_ylabel("Loss", fontsize=20)
    ax.tick_params(axis = "both", labelsize = 10)
    ax.set_title("Data size MSE", fontsize=25)
    ax.legend(loc="upper right", prop={'size': 20})
    ax.set_ylim(0,2e-4)
    plt.savefig("Data_size_mse.png")
    plt.close()

    """

    """

    sobol_300_loss, sobol_300_model = mg.NN_mc_model_1(sobol_set_300, "loss_history", "sobol_set_300", 5, 500, "normal", "False", "standardize", False, None, True)

    sobol_300_normalize_loss, sobol_300_normalize_model = mg.NN_mc_model_1(sobol_set_300, "loss_history", "sobol_300_normalize", 5, 500, "normal", "False", "normalize", False, None, True)

    test_set = random_input[:2, :]
    test_input = scaler.transform(test_set)
    test_imp = random_imp[:2, :]
    sobol_300_model.predict(test_set)
    some_test = np.array((-2,-2,-1,0,0,0,0))
    some_test = np.reshape(some_test, (1,-1))
    sobol_300_model.predict(some_test)
    some_model2 = load_model("Models5/loss_history/sobol_set_200_5_500.h5")

    sobol2_300_loss, sobol2_300_score = mg.NN_mc_model_1(sobol_set_300, "loss_history", "sobol2_set_300", 1, 50, "normal", "False", "standardize", False, None, True)



    fig = plt.figure(figsize=(10, 10), dpi = 200)
    ax = fig.add_subplot(111)
    ax.plot(sobol_300_loss.history["loss"], label = "Loss")
    ax.plot(sobol_300_loss.history["val_loss"], label = "Val Loss")
    ax.set_xlabel("Data set size", fontsize=20)
    ax.set_ylabel("Loss", fontsize=20)
    ax.tick_params(axis = "both", labelsize = 10)
    ax.set_title("Data size MSE", fontsize=25)
    ax.legend(loc="upper right", prop={'size': 20})
    ax.set_ylim(0,2e-2)
    plt.savefig("test.png")
    plt.close()

    random_input = np.loadtxt("Data/random_input_279936.csv", delimiter=",")
    random_imp = np.loadtxt("Data/random_imp_279936.csv", delimiter=",")
    test_index = np.random.choice(np.arange(len(random_imp)),size=30000, replace=False)

    models = glob.glob("Models5/loss_history/*.h5")

    models_mse = mt.model_test_set(models, random_input[test_index, :], random_imp[test_index, :])

    data_set_list = []
    sobol_set_list = []

    for some_list in models_mse:
        if (some_list[0].find("data") != -1):
            data_set_list.append((some_list[0][:8], 
            int(some_list[0][9:some_list[0].rfind("_")-2]),
            some_list[1]))
        else:
            sobol_set_list.append((some_list[0][:9], 
            int(some_list[0][10:some_list[0].rfind("_")-2]),
            some_list[1]))

    data_set_list.sort(key = lambda x: x[1])
    data_set_x = [sublist[2] for sublist in data_set_list]
    data_set_y = [sublist[1] for sublist in data_set_list]
    sobol_set_list.sort(key = lambda x: x[1])
    sobol_set_x = [sublist[2] for sublist in sobol_set_list]
    sobol_set_y = [sublist[1] for sublist in sobol_set_list]

    fig = plt.figure(figsize=(10, 10), dpi = 200)
    ax = fig.add_subplot(111)
    ax.plot(data_set_y, data_set_x, label = "Random from Sobol")
    ax.plot(sobol_set_y, sobol_set_x, label = "Sobol")
    ax.set_xlabel("Data set size", fontsize=20)
    ax.set_ylabel("Loss", fontsize=20)
    ax.tick_params(axis = "both", labelsize = 10)
    ax.set_title("Data size MSE", fontsize=25)
    ax.legend(loc="upper right", prop={'size': 20})
    ax.set_ylim(0,2e-4)
    plt.savefig("Data_size_mse.png")
    plt.close()
"""