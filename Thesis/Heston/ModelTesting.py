import numpy as np
import itertools
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
import glob
from sklearn.metrics import mean_squared_error as mse

from Thesis.Heston import AndersenLake as al, HestonModel as hm, DataGeneration as dg
from Thesis.misc import VanillaOptions as vo

def hard_case() -> np.ndarray:
    # Model inputs for test
    # Forward
    forward = 112.5

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

def find_nth_back(input_string : str, keyword : str, n : int):
    start = input_string.rfind("_")
    while start >= 0 and n > 1:
        start = input_string.rfind("_", 0, start)
        n -= 1
    return start

Grid_vs_sobol = glob.glob("Models2/grid_vs_sobol/*.h5")

Heston_single = glob.glob("Models2/Heston_single/*.h5")

Heston_non_normal = glob.glob("Models2/Heston_non_normal/sobol_200*.h5")

Heston_single_compare = Heston_single + Heston_non_normal

Heston_normal = glob.glob("Models2/Heston/sobol_200*.h5")

Heston_normal_compare = Heston_non_normal + Heston_normal

Heston_non_normal_tanh = glob.glob("Models2/Heston_non_normal_tanh/sobol_200*.h5")

Heston_non_normal_mix = glob.glob("Models2/Heston_non_normal_mix/sobol_200*.h5")

Heston_activation_compare = Heston_non_normal + Heston_non_normal_tanh + Heston_non_normal_mix

std_vs_nor = glob.glob("Models2/stardard_vs_normal/*.h5")
std_vs_nor_mix = glob.glob("Models2/standard_vs_normal_mix/*.h5")
std_vs_nor_tanh = glob.glob("Models2/standard_vs_normal_tanh/*.h5")

price_vs_imp = glob.glob("Models2/price_vs_imp/*.h5")

all_vs_filter = glob.glob("Models2/all_vs_filter/*.h5")

models = glob.glob("Models2/Heston/*.h5")

def model_testing_plot(model_list : list, plot_title : str, some_input : np.ndarray, option : np.ndarray):
    model_class = hm.HestonClass(some_input[0, 0], some_input[0, 1], some_input[0, 2], some_input[0, 3], some_input[0, 4], some_input[0, 5], some_input[0, 6])
    ending_list = [some_list[-4:-3] for some_list in model_list]
    some_option_list = np.array([])
    for some_option in option:
        some_option_list = np.append(some_option_list, vo.EUCall(some_option[0], some_option[1]))
    benchmark_price, benchmark = dg.calc_imp_vol(some_input[0], some_option_list)

    benchmark_price, benchmark = dg.calc_imp_vol(some_input[0], some_option_list)
    fig = plt.figure(figsize=(30, 10), dpi = 200)
    imp_ax = fig.add_subplot(121, projection='3d')
    error_ax = fig.add_subplot(122, projection='3d')

    no_models = len(model_list)
    color=iter(plt.cm.rainbow(np.linspace(0,1,no_models)))
    x = option[:,0]
    y = option[:,1]
    mse_list = []
    j = 0
    for model_string in model_list:    
        model = load_model(model_string)
        norm_feature = joblib.load(model_string[:-5]+"_norm_features_"+ending_list[j]+".pkl")
        try:
            norm_labels = joblib.load(model_string[:-5]+"_norm_labels_"+ending_list[j]+".pkl")
        except:
            normal_out = False
        else:
            normal_out = True

        if (model_string.find("Single") != -1 or model_string.find("single") != -1): # single output
            predictions = np.zeros(np.shape(option)[0])
            for i in range(np.shape(option)[0]):
                test_single_input = np.concatenate((some_input, option[i]), axis=None)
                test_single_input = np.reshape(test_single_input, (1, -1))
                if normal_out:
                    predictions[i] = norm_labels.inverse_transform(model.predict(norm_feature.transform(test_single_input)))
                else:
                    predictions[i] = model.predict(norm_feature.transform(test_single_input))
        else: # we have a grid
            if normal_out:
                predictions = norm_labels.inverse_transform(model.predict(norm_feature.transform(some_input)))[0]
            else:
                predictions = model.predict(norm_feature.transform(some_input))[0]

        # if prices, calc imp vol
        if (model_string.find("Price") != -1 or model_string.find("price") != -1 ):
            imp_vol_predictions = np.zeros(np.shape(predictions))
            for i in range(np.shape(predictions)[0]):
                imp_vol_predictions[i] = model_class.impVol(predictions[i], some_option_list[i])
            predictions = imp_vol_predictions
        c = next(color)

        z = predictions
        name = model_string[model_string.find("/")+1:]
        imp_ax.plot_trisurf(x, y, z, alpha = 0.5, label = name, color = c)
        mse_list.append((name, mse(predictions, benchmark)))
        error_ax.plot_trisurf(x, y, z - benchmark, alpha = 0.5, label = name, color = c)

        j += 1
    
    imp_ax.plot_trisurf(x, y, benchmark, color = "black", alpha = 0.5, label = "benchmark")

    imp_ax.set_ylabel("Strike")
    imp_ax.set_xlabel("Time to maturity")
    imp_ax.set_title("Implied volatility")

    error_ax.set_ylabel("Strike")
    error_ax.set_xlabel("Time to maturity")
    error_ax.set_title("Error")

    handles, labels = imp_ax.get_legend_handles_labels()
    for i in range(len(handles)):
        handles[i]._facecolors2d = handles[i]._facecolors3d 
        handles[i]._edgecolors2d = handles[i]._edgecolors3d 

    fig.subplots_adjust(top=0.95, left=0.1, right=0.95, bottom=0.3)
    fig.legend(handles, labels, loc="lower center", ncol = 4, fontsize=15)
    fig.suptitle(plot_title)
    plt.savefig("Plots2/"+plot_title+".png")
    plt.close()
    return mse_list

def model_testing(model_list : list, plot_title : str, some_input : np.ndarray, option : np.ndarray):
    model_class = hm.HestonClass(some_input[0, 0], some_input[0, 1], some_input[0, 2], some_input[0, 3], some_input[0, 4], some_input[0, 5], some_input[0, 6])
    ending_list = [some_list[-4:-3] for some_list in model_list]
    some_option_list = np.array([])
    for some_option in option:
        some_option_list = np.append(some_option_list, vo.EUCall(some_option[0], some_option[1]))
    benchmark_price, benchmark = dg.calc_imp_vol(some_input[0], some_option_list)

    benchmark_price, benchmark = dg.calc_imp_vol(some_input[0], some_option_list)
    fig = plt.figure(figsize=(30, 10), dpi = 200)
    imp_ax = fig.add_subplot(121, projection='3d')
    error_ax = fig.add_subplot(122, projection='3d')

    no_models = len(model_list)
    color=iter(plt.cm.rainbow(np.linspace(0,1,no_models)))
    x = option[:,0]
    y = option[:,1]
    mse_list = []
    j = 0
    for model_string in model_list:    
        model = load_model(model_string)
        model_folder = model_string[:model_string.rfind("/") + 1]
        norm_folder = "Models3/norms/"
        ### Check if model includes scaling
        if (model_string.find("/scaling") != -1):
            normal_out = True
            norm_labels = joblib.load(norm_folder+"norm_labels.pkl")

        if (model_string.find("price") != -1):
            norm_feature = joblib.load(norm_folder+"norm_feature_price.pkl")
        elif (model_string.find("grid_vs_sobol") != -1):
            if (model_string.find("sobol") != -1):
                norm_feature = joblib.load(norm_folder+"norm_feature_wide.pkl")
            else:
                norm_feature = joblib.load(norm_folder+"norm_feature_grid.pkl")
        elif (model_string.find("single") != -1):
            norm_feature = joblib.load(norm_folder+"norm_feature_single.pkl")
        else:
            norm_feature = joblib.load(norm_folder+"norm_feature.pkl")

        if (model_string.find("Single") != -1 or model_string.find("single") != -1): # single output
            predictions = np.zeros(np.shape(option)[0])
            for i in range(np.shape(option)[0]):
                test_single_input = np.concatenate((some_input, option[i]), axis=None)
                test_single_input = np.reshape(test_single_input, (1, -1))
                if normal_out:
                    predictions[i] = norm_labels.inverse_transform(model.predict(norm_feat.transform(test_single_input)))
                else:
                    predictions[i] = model.predict(norm_feat.transform(test_single_input))
        else: # we have a grid
            if normal_out:
                predictions = norm_labels.inverse_transform(model.predict(norm_feat.transform(some_input)))[0]
            else:
                predictions = model.predict(norm_feat.transform(some_input))[0]

        # if prices, calc imp vol
        if (model_string.find("Price") != -1 or model_string.find("price") != -1 ):
            imp_vol_predictions = np.zeros(np.shape(predictions))
            for i in range(np.shape(predictions)[0]):
                imp_vol_predictions[i] = model_class.impVol(predictions[i], some_option_list[i])
            predictions = imp_vol_predictions
        c = next(color)

        z = predictions
        name = model_string[model_string.find("/")+1:]
        imp_ax.plot_trisurf(x, y, z, alpha = 0.5, label = name, color = c)
        mse_list.append((name, mse(predictions, benchmark)))
        error_ax.plot_trisurf(x, y, z - benchmark, alpha = 0.5, label = name, color = c)

        j += 1
    
    imp_ax.plot_trisurf(x, y, benchmark, color = "black", alpha = 0.5, label = "benchmark")

    imp_ax.set_ylabel("Strike")
    imp_ax.set_xlabel("Time to maturity")
    imp_ax.set_title("Implied volatility")

    error_ax.set_ylabel("Strike")
    error_ax.set_xlabel("Time to maturity")
    error_ax.set_title("Error")

    handles, labels = imp_ax.get_legend_handles_labels()
    for i in range(len(handles)):
        handles[i]._facecolors2d = handles[i]._facecolors3d 
        handles[i]._edgecolors2d = handles[i]._edgecolors3d 

    fig.subplots_adjust(top=0.95, left=0.1, right=0.95, bottom=0.3)
    fig.legend(handles, labels, loc="lower center", ncol = 4, fontsize=15)
    fig.suptitle(plot_title)
    plt.savefig("Plots2/"+plot_title+".png")
    plt.close()
    return mse_list

def generate_barh_error(error_list : list, name : str):
    error_list.sort(key = lambda x: x[1])
    bar_fig = plt.figure(figsize=(10, 20), dpi = 200)
    bar_ax = bar_fig.add_subplot(111)
    labels, values = zip(*error_list)
    y_pos = np.arange(len(labels))
    bar_ax.barh(y_pos, values)
    bar_ax.set_yticks(y_pos)
    bar_ax.set_yticklabels(labels)
    bar_ax.invert_yaxis()
    bar_fig.suptitle("MSE with benchmark, "+name)
    plt.tight_layout()
    bar_fig.subplots_adjust(top=0.95, bottom=0.1)
    plt.savefig("Plots2/"+name+".png")
    plt.close()

def generate_bar_error(error_list : list, name : str):
    error_list.sort(key = lambda x: x[1])
    bar_fig = plt.figure(figsize=(20, 10), dpi = 200)
    bar_ax = bar_fig.add_subplot(111)
    labels, values = zip(*error_list)
    x_pos = np.arange(len(labels))
    bar_ax.bar(x_pos, values)
    bar_ax.set_xticks(x_pos)
    bar_ax.set_xticklabels(labels, rotation=90)
    bar_fig.suptitle("MSE with benchmark, "+name)
    plt.tight_layout()
    plt.savefig("Plots2/"+name+".png")
    plt.close()

def generate_plots(model_list : list, plot_title: str):
    easy_model_mse = model_testing_plot(model_list, plot_title+"_easy", easy_case(), option_input())
    generate_bar_error(easy_model_mse, plot_title+"_mse_easy")
    hard_model_mse = model_testing_plot(model_list, plot_title+"_hard", hard_case(), option_input())
    generate_bar_error(hard_model_mse, plot_title+"_mse_hard")

generate_plots(std_vs_nor, "standard_vs_normal")
generate_plots(std_vs_nor_mix, "standard_vs_normal_mix")
generate_plots(std_vs_nor_tanh, "standard_vs_normal_tanh")

generate_plots(price_vs_imp, "price_vs_imp")

generate_plots(all_vs_filter, "all_vs_filter")

### Using 200k sobol sets
price_imp_models = glob.glob("Models3/price_vs_imp/*.h5")
standard_normal_models = glob.glob("Models3/stardard_vs_normal/*.h5")
standard_normal_tanh_models = glob.glob("Models3/stardard_vs_normal_tanh/*.h5")
standard_normal_mix_models = glob.glob("Models3/stardard_vs_normal_mix/*.h5")

### Using single sets, from 200k sobol
single_models = glob.glob("Models3/single/*.h5")

### Using grid set, 279936 sets
grid_models = glob.glob("Models3/grid_vs_sobol/standard*.h5")
sobol_grid_models = glob.glob("Models3/grid_vs_sobol/sobol*.h5")
