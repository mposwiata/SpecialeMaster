import numpy as np
import itertools
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
import glob
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam
import os
import itertools
import pickle

from Thesis.Heston import AndersenLake as al, HestonModel as hm, DataGeneration as dg, ModelGenerator as mg
from Thesis.misc import VanillaOptions as vo

def hard_case() -> np.ndarray:
    # Model inputs for test
    # Forward
    forward = 150

    # vol
    vol = 0.01

    # kappa
    kappa = 0.1

    # theta
    theta = 0.01

    # epsilon
    epsilon = 2

    # rho
    rho = 0.8

    # rate
    rate = 0.05

    # Dataframe for NN with multiple outputs
    test_input = np.array((forward, vol, kappa, theta, epsilon, rho, rate))
    test_input = np.reshape(test_input, (1, 7))

    return test_input

def easy_case() -> np.ndarray:
    # Model inputs for test
    # Forward
    forward = 100

    # vol
    vol = 0.04

    # kappa
    kappa = 2

    # theta
    theta = 0.04

    # epsilon
    epsilon = 0.5

    # rho
    rho = -0.7

    # rate
    rate = 0.05

    # Dataframe for NN with multiple outputs
    test_input = np.array((forward, vol, kappa, theta, epsilon, rho, rate))
    test_input = np.reshape(test_input, (1, 7))

    return test_input

def option_input() -> np.ndarray:
    # Which options to test
    # Maturity
    maturity = np.linspace(start = 0.01, stop = 2, num = 5)

    # strike
    no_strikes = 5
    strike = np.linspace(start = 75, stop = 125, num = 5)

    option_input = np.array(list(itertools.product(maturity, strike))) # different option combinations
    return option_input

def model_testing2(model_list : list, plot_title : str) -> list:
    some_easy_case = easy_case()
    some_hard_case = hard_case()
    option = option_input()
    
    model_class_easy = hm.HestonClass(some_easy_case[0, 0], some_easy_case[0, 1], some_easy_case[0, 2], some_easy_case[0, 3], some_easy_case[0, 4], some_easy_case[0, 5], some_easy_case[0, 6])
    model_class_hard = hm.HestonClass(some_hard_case[0, 0], some_hard_case[0, 1], some_hard_case[0, 2], some_hard_case[0, 3], some_hard_case[0, 4], some_hard_case[0, 5], some_hard_case[0, 6])
    some_option_list = np.array([])
    for some_option in option:
        some_option_list = np.append(some_option_list, vo.EUCall(some_option[0], some_option[1]))
    benchmark_price_easy, benchmark_easy = dg.calc_imp_vol(some_easy_case[0], some_option_list)
    benchmark_price_hard, benchmark_hard = dg.calc_imp_vol(some_hard_case[0], some_option_list)

    fig = plt.figure(figsize=(30, 20), dpi = 200)
    imp_ax_easy = fig.add_subplot(221, projection='3d')
    error_ax_easy = fig.add_subplot(222, projection='3d')
    imp_ax_hard = fig.add_subplot(223, projection='3d')
    error_ax_hard = fig.add_subplot(224, projection='3d')

    no_models = len(model_list)
    color=iter(plt.cm.gist_rainbow(np.linspace(0,1,no_models)))
    x = option[:,0]
    y = option[:,1]
    mse_list = []
    for model_string in model_list:
        some_easy_case = easy_case()
        some_hard_case = hard_case()
        model_string = ' '.join(glob.glob("Models5/*/"+model_string))
        model = load_model(model_string)
        model_folder = model_string[:model_string.rfind("/") + 1]
        if os.path.exists(model_folder+"/norm_feature.pkl"):
            norm_feature = joblib.load(model_folder+"norm_feature.pkl")
            normal_in = True
        else:
            normal_in = False
        
        if os.path.exists(model_folder+"/norm_labels.pkl"):
            norm_labels = joblib.load(model_folder+"norm_labels.pkl")
            normal_out = True
        else:
            normal_out = False

        if (model_string.find("Single") != -1 or model_string.find("single") != -1): # single output
            predictions_easy = np.zeros(np.shape(option)[0])
            predictions_hard = np.zeros(np.shape(option)[0])
            for i in range(np.shape(option)[0]):
                test_single_input_easy = np.concatenate((some_easy_case, option[i]), axis=None)
                test_single_input_easy = np.reshape(test_single_input_easy, (1, -1))
                test_single_input_hard = np.concatenate((some_hard_case, option[i]), axis=None)
                test_single_input_hard = np.reshape(test_single_input_hard, (1, -1))
                if normal_in:
                    test_single_input_easy = norm_feature.transform(test_single_input_easy)
                    test_single_input_hard = norm_feature.transform(test_single_input_hard)
                if normal_out:
                    predictions_easy[i] = norm_labels.inverse_transform(model.predict(test_single_input_easy))
                    predictions_hard[i] = norm_labels.inverse_transform(model.predict(test_single_input_hard))
                else:
                    predictions_easy[i] = model.predict(test_single_input_easy)
                    predictions_hard[i] = model.predict(test_single_input_hard)
        else: # we have a grid
            if normal_in:
                some_easy_case = norm_feature.transform(some_easy_case)
                some_hard_case = norm_feature.transform(some_hard_case)
            if normal_out:
                predictions_easy = norm_labels.inverse_transform(model.predict(some_easy_case))[0]
                predictions_hard = norm_labels.inverse_transform(model.predict(some_hard_case))[0]
            else:
                predictions_easy = model.predict(some_easy_case)[0]
                predictions_hard = model.predict(some_hard_case)[0]

        # if prices, calc imp vol
        if (model_string.find("Price") != -1 or model_string.find("price") != -1 ):
            imp_vol_predictions_easy = np.zeros(np.shape(predictions_easy))
            imp_vol_predictions_hard = np.zeros(np.shape(predictions_hard))
            for i in range(np.shape(predictions_easy)[0]):
                imp_vol_predictions_easy[i] = model_class_easy.impVol(predictions_easy[i], some_option_list[i])
                imp_vol_predictions_hard[i] = model_class_hard.impVol(predictions_hard[i], some_option_list[i])
            predictions_easy = imp_vol_predictions_easy
            predictions_hard = imp_vol_predictions_hard
            
        c = next(color)

        z_easy = predictions_easy
        z_hard = predictions_hard
        name = model_string[model_string.find("/")+1:]
        imp_ax_easy.plot_trisurf(x, y, z_easy, alpha = 0.5, label = name, color = c)
        imp_ax_hard.plot_trisurf(x, y, z_hard, alpha = 0.5, label = name, color = c)
        predictions = np.concatenate((predictions_easy, predictions_hard))
        benchmark = np.concatenate((benchmark_easy, benchmark_hard))
        mse_list.append((name, mse(predictions, benchmark)))
        error_ax_easy.plot_trisurf(x, y, z_easy - benchmark_easy, alpha = 0.5, label = name, color = c)
        error_ax_hard.plot_trisurf(x, y, z_hard - benchmark_hard, alpha = 0.5, label = name, color = c)
    
    imp_ax_easy.plot_trisurf(x, y, benchmark_easy, color = "black", alpha = 0.5, label = "benchmark")

    imp_ax_easy.set_ylabel("Strike")
    imp_ax_easy.set_xlabel("Time to maturity")
    imp_ax_easy.set_title("Implied volatility, easy case")

    error_ax_easy.set_ylabel("Strike")
    error_ax_easy.set_xlabel("Time to maturity")
    error_ax_easy.set_title("Error")

    imp_ax_hard.plot_trisurf(x, y, benchmark_hard, color = "black", alpha = 0.5, label = "benchmark")

    imp_ax_hard.set_ylabel("Strike")
    imp_ax_hard.set_xlabel("Time to maturity")
    imp_ax_hard.set_title("Implied volatility, hard case")

    error_ax_hard.set_ylabel("Strike")
    error_ax_hard.set_xlabel("Time to maturity")
    error_ax_hard.set_title("Error")

    handles, labels = imp_ax_easy.get_legend_handles_labels()
    for i in range(len(handles)):
        handles[i]._facecolors2d = handles[i]._facecolors3d 
        handles[i]._edgecolors2d = handles[i]._edgecolors3d 

    fig.subplots_adjust(top=0.95, left=0.1, right=0.95, bottom=0.2)
    fig.legend(handles, labels, loc="lower center", ncol = 4, fontsize=15)
    fig.suptitle(plot_title, fontsize=20)
    plt.savefig("Final_plots/"+plot_title.replace(" ", "_")+".png")
    plt.close()
    return mse_list

def model_test_set(model_list : list, X_test : np.ndarray, Y_test : np.ndarray, Y_test_price : np.ndarray = None) -> list:
    mse_list = []
    for model_string in model_list:
        if (model_string.find("price") != -1):
            y_test_loop = Y_test_price
        else:
            y_test_loop = Y_test
       
        x_test_loop = X_test

        if (model_string.find("single") != -1):
            x_test_loop, y_test_loop = mg.transform_single(x_test_loop, y_test_loop)
        elif (model_string.find("mat") != -1):
            x_test_loop, y_test_loop = mg.transform_mat(x_test_loop, y_test_loop)
        
        if ((model_string.find("benchmark_include") != -1) or (model_string.find("price_include") != -1)):
            index = np.all(y_test_loop != -1, axis = 1)
        else:
            index = np.all(y_test_loop > 0, axis = 1)

        x_test_loop = x_test_loop[index, :]
        y_test_loop = y_test_loop[index, :]

        model = load_model(model_string)
        model_folder = model_string[:model_string.rfind("/") + 1]
        if os.path.exists(model_folder+"/norm_feature.pkl"):
            norm_feature = joblib.load(model_folder+"norm_feature.pkl")
            x_test_loop = norm_feature.transform(x_test_loop)
        if os.path.exists(model_folder+"/norm_labels.pkl"):
            norm_labels = joblib.load(model_folder+"norm_labels.pkl")
            y_test_loop = norm_labels.transform(y_test_loop)

        name = model_string[model_string.rfind("/")+1:]
        score = model.evaluate(x_test_loop, y_test_loop, verbose=0)
        mse(model.predict(x_test_loop), y_test_loop)
        mse_list.append((name, score))
    
    return mse_list

def generate_bar_error(error_list : list, name : str):
    bar_fig = plt.figure(figsize=(20, 10), dpi = 200)
    bar_ax = bar_fig.add_subplot(111)
    labels, values = zip(*error_list)
    labels = [l[11:-3].replace("_", ", ") for l in labels]
    x_pos = np.arange(len(labels))
    bar_ax.bar(x_pos, values)
    bar_ax.set_xticks(x_pos)
    bar_ax.set_xticklabels(labels, rotation=45, fontsize=15, ha = "right")
    bar_fig.suptitle("MSE, "+name)
    plt.tight_layout()
    plt.savefig("Final_plots/"+name.replace(" ", "_")+"_mse.png")
    plt.close()

def generate_plots(model_list : list, plot_title: str):
    mse = model_testing2(model_list, plot_title)
    generate_bar_error(mse, plot_title)

if __name__ == "__main__":
    sama_param_models = glob.glob("Models5/same_param/*.h5")

    train_index, test_index = mg.load_index(200000)
    model_input = np.loadtxt("Data/benchmark_input.csv", delimiter = ",")
    imp_vol = np.loadtxt("Data/benchmark_imp.csv", delimiter=",")
    price = np.loadtxt("Data/benchmark_price.csv", delimiter=",")

    X_test = model_input[test_index, :]
    Y_test = imp_vol[test_index, :]
    Y_test_price = price[test_index, :]

    same_param_mse = model_test_set(sama_param_models, X_test, Y_test, Y_test_price)
    same_param_mse.sort(key = lambda x: int(x[0][11:x[0].rfind("_")]))
    errors = ['{0:.7f}'.format(round(s[1], 7)) for s in same_param_mse]
    generate_bar_error(same_param_mse, "Same no. of parameters")