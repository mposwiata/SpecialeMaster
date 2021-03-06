import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
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
    forward = 100

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

def model_testing(model_list : list, plot_title : str, easy_case : np.ndarray, hard_case : np.ndarray, option : np.ndarray) -> list:
    model_class_easy = hm.HestonClass(easy_case[0, 0], easy_case[0, 1], easy_case[0, 2], easy_case[0, 3], easy_case[0, 4], easy_case[0, 5], easy_case[0, 6])
    model_class_hard = hm.HestonClass(hard_case[0, 0], hard_case[0, 1], hard_case[0, 2], hard_case[0, 3], hard_case[0, 4], hard_case[0, 5], hard_case[0, 6])
    some_option_list = np.array([])
    for some_option in option:
        some_option_list = np.append(some_option_list, vo.EUCall(some_option[0], some_option[1]))
    benchmark_price_easy, benchmark_easy = dg.calc_imp_vol(easy_case[0], some_option_list)
    benchmark_price_hard, benchmark_hard = dg.calc_imp_vol(hard_case[0], some_option_list)

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
        model = load_model(model_string)
        model_folder = model_string[:model_string.rfind("/") + 1]
        norm_folder = "Models4/norms/"
        ### Check if model includes scaling
        if ((model_string.find("output_scaling") != -1) or ((model_string.find("final") != -1 and model_string.find("final2") == -1 ) or (model_string.find("price_standard") != -1))):
            if (model_string.find("price") != -1):
                norm_labels = joblib.load(norm_folder+"norm_labels_price.pkl")
            else: 
                norm_labels = joblib.load(norm_folder+"norm_labels.pkl")
            normal_out = True
        else:
            normal_out = False

        if (model_string.find("price") != -1):
            norm_feature = joblib.load(norm_folder+"norm_feature_price.pkl")
        elif (model_string.find("grid_vs_sobol") != -1):
            if (model_string.find("sobol") != -1):
                norm_feature = joblib.load(norm_folder+"norm_feature_wide.pkl")
            else:
                norm_feature = joblib.load(norm_folder+"norm_feature_grid.pkl")
        elif (model_string.find("single") != -1):
            norm_feature = joblib.load(norm_folder+"norm_feature_single.pkl")
        elif (model_string.find("standard") != -1 or ((model_string.find("final") != -1 and model_string.find("final3") == -1 ))):
            norm_feature = joblib.load(norm_folder+"standard_features.pkl")
        else:
            norm_feature = joblib.load(norm_folder+"norm_feature.pkl")

        if (model_string.find("Single") != -1 or model_string.find("single") != -1): # single output
            predictions_easy = np.zeros(np.shape(option)[0])
            predictions_hard = np.zeros(np.shape(option)[0])
            for i in range(np.shape(option)[0]):
                test_single_input_easy = np.concatenate((easy_case, option[i]), axis=None)
                test_single_input_easy = np.reshape(test_single_input_easy, (1, -1))
                test_single_input_hard = np.concatenate((hard_case, option[i]), axis=None)
                test_single_input_hard = np.reshape(test_single_input_hard, (1, -1))
                if normal_out:
                    predictions_easy[i] = norm_labels.inverse_transform(model.predict(norm_feature.transform(test_single_input_easy)))
                    predictions_hard[i] = norm_labels.inverse_transform(model.predict(norm_feature.transform(test_single_input_hard)))
                else:
                    predictions_easy[i] = model.predict(norm_feature.transform(test_single_input_easy))
                    predictions_hard[i] = model.predict(norm_feature.transform(test_single_input_hard))
        else: # we have a grid
            if normal_out:
                predictions_easy = norm_labels.inverse_transform(model.predict(norm_feature.transform(easy_case)))[0]
                predictions_hard = norm_labels.inverse_transform(model.predict(norm_feature.transform(hard_case)))[0]
            else:
                predictions_easy = model.predict(norm_feature.transform(easy_case))[0]
                predictions_hard = model.predict(norm_feature.transform(hard_case))[0]

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
        name = model_string[model_string.rfind("/")+1:]
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
    fig.legend(handles, labels, loc="lower center", ncol = 5, fontsize=15)
    fig.suptitle(plot_title, fontsize=20)
    plt.savefig("Plots2/"+plot_title.replace(" ", "_")+".png")
    plt.close()
    return mse_list

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
        name = model_string[model_string.rfind("/")+1:]
        imp_ax_easy.plot_trisurf(x, y, z_easy, alpha = 0.5, label = name[:-3], color = c)
        imp_ax_hard.plot_trisurf(x, y, z_hard, alpha = 0.5, label = name[:-3], color = c)
        predictions = np.concatenate((predictions_easy, predictions_hard))
        benchmark = np.concatenate((benchmark_easy, benchmark_hard))
        mse_list.append((name, mse(predictions, benchmark)))
        error_ax_easy.plot_trisurf(x, y, z_easy - benchmark_easy, alpha = 0.5, label = name, color = c)
        error_ax_hard.plot_trisurf(x, y, z_hard - benchmark_hard, alpha = 0.5, label = name, color = c)
    
    imp_ax_easy.plot_trisurf(x, y, benchmark_easy, color = "black", alpha = 0.5, label = "benchmark")

    imp_ax_easy.set_ylabel("\n" + "Strike", linespacing=4, fontsize = 20)
    imp_ax_easy.set_xlabel("\n" + "Time to maturity", linespacing=4, fontsize = 20)
    imp_ax_easy.set_title("Implied volatility, easy case", fontsize = 25)
    imp_ax_easy.tick_params(labelsize = 15)

    error_ax_easy.set_ylabel("\n" + "Strike", linespacing=4, fontsize = 20)
    error_ax_easy.set_xlabel("\n" + "Time to maturity", linespacing=4, fontsize = 20)
    error_ax_easy.set_title("Error", fontsize = 25)
    error_ax_easy.tick_params(labelsize = 15)

    imp_ax_hard.plot_trisurf(x, y, benchmark_hard, color = "black", alpha = 0.5, label = "benchmark")

    imp_ax_hard.set_ylabel("\n" + "Strike", linespacing=4, fontsize = 20)
    imp_ax_hard.set_xlabel("\n" + "Time to maturity", linespacing=4, fontsize = 20)
    imp_ax_hard.set_title("Implied volatility, hard case", fontsize = 25)
    error_ax_easy.tick_params(labelsize = 15)

    error_ax_hard.set_ylabel("\n" + "Strike", linespacing=4, fontsize = 20)
    error_ax_hard.set_xlabel("\n" + "Time to maturity", linespacing=4, fontsize = 20)
    error_ax_hard.set_title("Error", fontsize = 25)
    error_ax_easy.tick_params(labelsize = 15)

    handles, labels = imp_ax_easy.get_legend_handles_labels()
    for i in range(len(handles)):
        handles[i]._facecolors2d = handles[i]._facecolors3d 
        handles[i]._edgecolors2d = handles[i]._edgecolors3d 

    fig.subplots_adjust(top=0.95, left=0.1, right=0.95, bottom=0.2)
    fig.legend(handles, labels, loc="lower center", ncol = 3, prop={'size': 30})
    fig.suptitle(plot_title, fontsize=30)
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
        normal_out = False
        if os.path.exists(model_folder+"/norm_labels.pkl"):
            norm_labels = joblib.load(model_folder+"norm_labels.pkl")
            #y_test_loop = norm_labels.transform(y_test_loop)
            normal_out = True
        name = model_string[model_string.rfind("/")+1:]
        if normal_out:
            predictions = norm_labels.inverse_transform(model.predict(x_test_loop))
        else:
            predictions = model.predict(x_test_loop)
        #score = model.evaluate(x_test_loop, y_test_loop, verbose=0)
        score = mse(predictions, y_test_loop)
        mse_list.append((name, score))
    
    return mse_list

def monte_carlo_testing(model_list : list, X_test : np.ndarray, Y_test : np.ndarray):
    norm_feature = joblib.load("Models4/Heston_input_scale.pkl")
    mse_list = []
    for model_string in model_list:
        x_test_loop = X_test
        y_test_loop = Y_test
        model = load_model(model_string)
        ### Check if model includes scaling
        if (model_string.find("price") != -1):
            norm_labels = joblib.load(model_string[:model_string.rfind("/")+1]+"price_scale.pkl")
            y_test_loop = norm_labels.transform(y_test_loop)

        x_test_loop = norm_feature.transform(x_test_loop)

        name = model_string[model_string.find("/")+1:]
        score = model.evaluate(x_test_loop, y_test_loop, verbose=0)

        mse_list.append((name, score))
    
    return mse_list

def generate_bar_error(error_list : list, name : str):
    error_list.sort(key = lambda x: x[1])
    tmp_list = []
    unique_vals = []
    for some_list in error_list:
        unique_vals.append(some_list[0][:some_list[0].rfind("_")-2])
        tmp_list.append((some_list[0][:some_list[0].rfind("_")-2], some_list[0][some_list[0].rfind("_")-1:], some_list[1]))
    unique_vals = list(set(unique_vals))
    colors = cm.Set2(np.linspace(0, 1, len(unique_vals)))
    
    bar_list = []
    color_list = []
    for some_list in tmp_list:
        color_list.append(colors[unique_vals.index(some_list[0])])
        bar_list.append((some_list[1][:-3].replace("_", ", "), some_list[2]))

    bar_fig = plt.figure(figsize=(20, 10), dpi = 200)
    bar_ax = bar_fig.add_subplot(111)
    labels, values = zip(*bar_list)
    x_pos = np.arange(len(labels))
    bar_ax.bar(x_pos, values, color = color_list)
    bar_ax.yaxis.offsetText.set_fontsize(15)
    bar_ax.tick_params(axis = "y", labelsize=15)
    bar_ax.set_xticks(x_pos)
    bar_ax.set_xticklabels(labels, rotation=45, fontsize=20, ha = "right")
    bar_ax.set_title("MSE, "+name,size=25)
    plt.tight_layout()
    handels = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(unique_vals))]
    plt.legend(handels, unique_vals, loc="upper left", prop={'size': 20})
    plt.savefig("Final_plots/"+name.replace(" ", "_")+"_mse.png")
    plt.close()

def generate_plots(model_list : list, plot_title: str):
    mse = model_testing2(model_list, plot_title)
    generate_bar_error(mse, plot_title)

if __name__ == "__main__":
    price_models = glob.glob("Models5/price*/*.h5")

    imp_model_keys = [
        "benchmark",
        "benchmark_include",
        "output_scaling",
        "output_scaling_normalize",
        "mix",
        "tanh",
        "standardize",
        "mix_standardize",
        "tanh_standardize",
        "non_input_scaling"
    ]
    imp_models = []
    for key in imp_model_keys:
        imp_models.append(glob.glob("Models5/"+key+"/*.h5"))
    
    imp_models = [item for sublist in imp_models for item in sublist]

    train_index, test_index = mg.load_index(200000)
    model_input = np.loadtxt("Data/benchmark_input.csv", delimiter = ",")
    imp_vol = np.loadtxt("Data/benchmark_imp.csv", delimiter=",")
    price = np.loadtxt("Data/benchmark_price.csv", delimiter=",")
    price_old = np.loadtxt("Data/benchmark_price_old.csv", delimiter=",")
    imp_vol_old = np.loadtxt("Data/benchmark_imp_old.csv", delimiter = ",")  

    X_test = model_input[test_index, :]
    Y_test = imp_vol[test_index, :]
    Y_test_old = imp_vol_old[test_index, :]
    Y_test_price = price[test_index, :]
    Y_test_price_old = price_old[test_index, :]

    price_mse = model_test_set(price_models, X_test, Y_test, Y_test_price)
    imp_mse = model_test_set(imp_models, X_test, Y_test, Y_test_price)

    with open("price_mse.pkl", "wb") as fp:   #Pickling
        pickle.dump(price_mse, fp)

    with open("imp_mse.pkl", "wb") as fp:   #Pickling
        pickle.dump(imp_mse, fp)

    with open("imp_mse.pkl", "rb") as fp:   #Pickling
        imp_mse = pickle.load(fp)

    with open("price_mse.pkl", "rb") as fp:   #Pickling
        price_mse = pickle.load(fp)

    evaluation_first_list = []
    for some_list in price_mse:
        evaluation_first_list.append(
            [some_list[0][:some_list[0].rfind("_")-2], some_list[0][some_list[0].rfind("_")-1:], some_list[1]]
        )
    
    ### Finding best models per group
    evaluation_first_list.sort(key = lambda x: x[0])
    group_by_model = itertools.groupby(evaluation_first_list, key = lambda x: x[0])

    top_first_models_list = []
    for key, group in group_by_model:
        some_list = list(group)
        some_list.sort(key = lambda x: x[2])
        top_first_models_list.append(some_list[0])

    ### Finding best models per setup
    evaluation_setup_list = []
    for some_list in price_mse:
        evaluation_setup_list.append(
            [some_list[0][:some_list[0].rfind("_")-2], some_list[0][some_list[0].rfind("_")-1:], some_list[1]]
        )
    evaluation_setup_list.sort(key = lambda x: x[1])
    group_by_network = itertools.groupby(evaluation_setup_list, key = lambda x: x[1])

    top_first_network_list = []
    for key, group in group_by_network:
        some_list = list(group)
        some_list.sort(key = lambda x: x[2])
        top_first_network_list.append(some_list[0])

    top_first_network_list.sort(key = lambda x: x[1])

    ### Random vs sobol vs grid
    data_type_model_keys = [
        "random_data2",
        "grid_sobol",
        "grid"
    ]
    data_type_models = []
    for key in data_type_model_keys:
        data_type_models.append(glob.glob("Models5/"+key+"/*.h5"))
    
    data_type_models = [item for sublist in data_type_models for item in sublist]

    random_train_index, random_test_index = mg.load_index(78125)
    random_input = np.loadtxt("Data/random_input_78125.csv", delimiter = ",")
    random_imp = np.loadtxt("Data/random_imp_78125.csv", delimiter = ",")
    grid_input = np.loadtxt("Data/grid_input.csv", delimiter = ",")
    grid_imp = np.loadtxt("Data/grid_imp.csv", delimiter = ",")

    data_type_mse = model_test_set(data_type_models, random_input[random_test_index, :], random_imp[random_test_index, :])
    data_type_mse_sobol = model_test_set(data_type_models, model_input[random_test_index, :], imp_vol[random_test_index, :])
    data_type_mse_grid = model_test_set(data_type_models, grid_input[random_test_index, :], grid_imp[random_test_index, :])
    data_type_mse.sort(key = lambda x: x[1])
    data_type_mse_sobol.sort(key = lambda x: x[1])
    data_type_mse_grid.sort(key = lambda x: x[1])
    generate_bar_error(data_type_mse[:15], "Model input on random test set")
    generate_bar_error(data_type_mse_sobol[:15], "Model input on sobol test set")
    generate_bar_error(data_type_mse_grid[:15], "Model input on grid test set")

    ### 200K
    ### Random vs sobol vs grid
    data_type_model_keys = [
        "random_data200000",
        "benchmark"
    ]
    data_type_models = []
    for key in data_type_model_keys:
        data_type_models.append(glob.glob("Models5/"+key+"/*.h5"))
    
    data_type_models = [item for sublist in data_type_models for item in sublist]
    random_input_200 = np.loadtxt("Data/random_input_200000.csv", delimiter = ",")
    random_imp_200 = np.loadtxt("Data/random_imp_200000.csv", delimiter = ",")

    data_type_200_mse = model_test_set(data_type_models, X_test, Y_test)
    data_type_200_random_mse = model_test_set(data_type_models, random_input_200[test_index, :], random_imp_200[test_index, :])
    data_type_200_mse.sort(key = lambda x: x[1])
    data_type_200_random_mse.sort(key = lambda x: x[1])

    combined_list = []
    first_run_keys = [
        "benchmark",
        "benchmark_include",
        "output_scaling",
        "output_scaling_normalize",
        "mix",
        "price",
        "price_include",
        "price_standardize",
        "price_output_standardize",
        "price_output_normalize",
        "tanh",
        "standardize",
        "mix_standardize",
        "tanh_standardize",
        "non_input_scaling",
        "standardize_single",
        "standardize_mat"
    ]

    for key in tmp_run:
        combined_list.append(glob.glob("Models5/"+key+"/*.h5"))

    combined_list = [item for sublist in combined_list for item in sublist]



    ### 100.000
    test_index_1 = np.loadtxt("Data/test_index_100000.csv", delimiter=",").astype(int)
    X_test_1 = np.loadtxt("Data/100000_input.csv", delimiter = ",")[test_index_1, :]
    Y_test_1 = np.loadtxt("Data/100000_imp.csv", delimiter = ",")[test_index_1, :]
    low_data_models = glob.glob("Models5/low_data/*.h5")
    low_data_models_mse = model_test_set(low_data_models, X_test_1, Y_test_1)
    low_data_models_mse.sort(key = lambda x: x[1])

    test_index_3 = np.loadtxt("Data/test_index_300000.csv", delimiter=",").astype(int)
    X_test_3 = np.loadtxt("Data/300000_input.csv", delimiter = ",")[test_index_3, :]
    Y_test_3 = np.loadtxt("Data/300000_imp.csv", delimiter = ",")[test_index_3, :]
    high_data_models = glob.glob("Models5/high_data/*.h5")
    high_data_models_mse = model_test_set(high_data_models, X_test_3, Y_test_3)
    high_data_models_mse.sort(key = lambda x: x[1])

    test_index_grid = np.loadtxt("Data/test_index_279936.csv", delimiter=",").astype(int)
    X_test_grid = np.loadtxt("Data/279936_input.csv", delimiter = ",")[test_index_grid, :]
    Y_test_grid = np.loadtxt("Data/279936_imp.csv", delimiter = ",")[test_index_grid, :]
    X_test_sobol_grid = np.loadtxt("Data/grid_input.csv", delimiter = ",")[test_index_grid, :]
    Y_test_sobol_grid = np.loadtxt("Data/grid_imp.csv", delimiter = ",")[test_index_grid, :]
    grid_models = glob.glob("Models5/grid/*.h5")
    sobol_models = glob.glob("Models5/sobol/*.h5")
    grid_models_mse = model_test_set(grid_models, X_test_grid, Y_test_grid)
    sobol_models_mse = model_test_set(sobol_models, X_test_grid, Y_test_grid)
    grid_models_mse_grid_inp = model_test_set(grid_models, X_test_sobol_grid, Y_test_sobol_grid)
    sobol_models_mse_grid_inp = model_test_set(sobol_models, X_test_sobol_grid, Y_test_sobol_grid)
    grid_models_mse_grid_inp.sort(key = lambda x: x[1])
    sobol_models_mse_grid_inp.sort(key = lambda x: x[1])
    generate_bar_error(grid_models_mse_grid_inp[:5]+sobol_models_mse_grid_inp[:5], "Model input generation")

    combined_mse = model_test_set(combined_list, X_test, Y_test, Y_test_price)
    for i in range(20):
        combined_mse.append(low_data_models_mse[i])
        combined_mse.append(high_data_models_mse[i])
        combined_mse.append(grid_models_mse[i])
        combined_mse.append(sobol_models_mse[i])

    with open("final_combined_mse.pkl", "wb") as fp:   #Pickling
        pickle.dump(combined_mse, fp)

    with open("final_combined_mse.pkl", "rb") as fp:   # Unpickling
        combined_mse = pickle.load(fp)

    combined_mse.sort(key = lambda x: -x[1])

    evaluation_first_list = []
    for some_list in combined_mse:
        evaluation_first_list.append(
            [some_list[0][:some_list[0].rfind("_")-2], some_list[0][some_list[0].rfind("_")-1:], '{0:.7f}'.format(round(some_list[1], 7))]
        )

    ### Models not converging
    non_convergence = ["output_scaling_4_1000.h5", "output_scaling_5_1000.h5"]
    model_testing2(non_convergence, "Non converging models")
    
    ### Finding best models per group
    evaluation_first_list.sort(key = lambda x: x[0])
    group_by_model = itertools.groupby(evaluation_first_list, key = lambda x: x[0])

    top_first_models_list = []
    for key, group in group_by_model:
        some_list = list(group)
        some_list.sort(key = lambda x: x[2])
        top_first_models_list.append(some_list[0])

    ### Finding best models per setup
    evaluation_setup_list = []
    for some_list in combined_mse:
        if not (some_list[0].find("price") != -1 or some_list[0].find("high_data") != -1 or \
            some_list[0].find("sobol") != -1 or some_list[0].find("standardize_mat") != -1 or \
            some_list[0].find("grid") != -1 or some_list[0].find("low_data") != -1 or \
            some_list[0].find("standardize_single") != -1):
            evaluation_setup_list.append(
                [some_list[0][:some_list[0].rfind("_")-2], some_list[0][some_list[0].rfind("_")-1:], '{0:.7f}'.format(round(some_list[1], 7))]
            )
    evaluation_setup_list.sort(key = lambda x: x[1])
    group_by_network = itertools.groupby(evaluation_setup_list, key = lambda x: x[1])

    top_first_network_list = []
    for key, group in group_by_network:
        some_list = list(group)
        some_list.sort(key = lambda x: x[2])
        top_first_network_list.append(some_list[0])

    top_first_network_list.sort(key = lambda x: x[1])

    models_for_evaluation = [
        "mix_standardize_5_1000.h5",
        "output_scaling_5_100.h5",
        "standardize_5_500.h5",
        "mix_3_1000.h5",
        "mix_standardize_2_500.h5",
        "mix_standardize_1_500.h5",
        "tanh_standardize_1_50.h5",
        "output_scaling_4_50.h5"
    ]
    model_testing2(models_for_evaluation, "Top implied volatility models")
    model_top_list = []
    for some_model in models_for_evaluation:
        model_top_list.append(' '.join(glob.glob("Models5/*/"+some_model)))

    top_mse = model_test_set(model_top_list, X_test, Y_test, Y_test_price)
    generate_bar_error(top_mse, "Top implied volatility models")

    price_plot = ["price_standardize_4_1000.h5"]
    generate_plots(price_plot, "Best price model")
    
    imp_plot = ["standardize_5_1000.h5"]
    generate_plots(imp_plot, "Best implied volatility model")

    ### Finding best models per setup, prices
    evaluation_setup_price_list = []
    for some_list in combined_mse:
        if some_list[0].find("price") != -1:
            evaluation_setup_price_list.append(
                [some_list[0][:some_list[0].rfind("_")-2], some_list[0][some_list[0].rfind("_")-1:], '{0:.7f}'.format(round(some_list[1], 7))]
            )
    evaluation_setup_price_list.sort(key = lambda x: x[1])
    group_by_network = itertools.groupby(evaluation_setup_price_list, key = lambda x: x[1])

    top_first_network_price_list = []
    for key, group in group_by_network:
        some_list = list(group)
        some_list.sort(key = lambda x: x[2])
        top_first_network_price_list.append(some_list[0])

    top_first_network_price_list.sort(key = lambda x: x[1])

    ### Price vs implied
    price_imp_dict = {
        "price" : ["price", "price_include", "price_standardize", "price_output_standardize", \
            "price_output_normalize"],
        "imp" : ["benchmark", "benchmark_inlcude", "output_scaling", "output_scaling_normalize", \
            "standardize", "non_input_scaling "]
    }

    price_models = []
    imp_models = []
    for some_list in price_mse + imp_mse:
        if (price_imp_dict["price"].count(some_list[0][:some_list[0].rfind("_")-2]) > 0):
            price_models.append(some_list)
        elif (price_imp_dict["imp"].count(some_list[0][:some_list[0].rfind("_")-2]) > 0):
            imp_models.append(some_list)
    price_models.sort(key = lambda x: x[1])
    imp_models.sort(key = lambda x: x[1])
    price_imp_models = []
    for i in range(5):
        price_imp_models.append(price_models[i][0])
        price_imp_models.append(imp_models[i][0])
    model_testing2(price_imp_models, "Price vs implied volatility")
    price_imp_mse = imp_models[0:5] + price_models[0:5]
    generate_bar_error(price_imp_mse, "Price vs implied volatility")

    ### Data filtering
    benchmark_models = []
    benchmark_include_models = []
    for some_list in price_mse + imp_mse:
        if (some_list[0][:some_list[0].rfind("_")-2] == "benchmark"):
            benchmark_models.append(some_list)
        elif (some_list[0][:some_list[0].rfind("_")-2] == "benchmark_include"):
            benchmark_include_models.append(some_list)
    benchmark_models.sort(key = lambda x: x[1])
    benchmark_include_models.sort(key = lambda x : x[1])
    data_filter_models = []
    for i in range(5):
        data_filter_models.append(benchmark_models[i][0])
        data_filter_models.append(benchmark_include_models[i][0])
    #model_testing2(data_filter_models, "Data filtering")
    generate_bar_error(benchmark_models[0:5] + benchmark_include_models[0:5], "Data filtering")

    ### Input scaling
    benchmark_models = []
    standardize_models = []
    non_input_scaling_models = []
    for some_list in price_mse + imp_mse:
        if (some_list[0][:some_list[0].rfind("_")-2] == "benchmark"):
            benchmark_models.append(some_list)
        elif (some_list[0][:some_list[0].rfind("_")-2] == "standardize"):
            standardize_models.append(some_list)
        elif (some_list[0][:some_list[0].rfind("_")-2] == "non_input_scaling"):
            non_input_scaling_models.append(some_list)
    benchmark_models.sort(key = lambda x: x[1])
    standardize_models.sort(key = lambda x : x[1])
    non_input_scaling_models.sort(key = lambda x: x[1])
    generate_bar_error(benchmark_models[0:5] + standardize_models[0:5] + non_input_scaling_models[0:5], "Input scaling")
    input_scaling_models = []
    for i in range(5):
        input_scaling_models.append(benchmark_models[i][0])
        input_scaling_models.append(standardize_models[i][0])
        input_scaling_models.append(non_input_scaling_models[i][0])
    #model_testing2(input_scaling_models, "Input scaling")

    ### Output scaling
    benchmark_models = []
    output_scaling_models = []
    output_scaling_normalize_models = []
    for some_list in price_mse + imp_mse:
        if (some_list[0][:some_list[0].rfind("_")-2] == "benchmark"):
            benchmark_models.append(some_list)
        elif (some_list[0][:some_list[0].rfind("_")-2] == "output_scaling"):
            output_scaling_models.append(some_list)
        elif (some_list[0][:some_list[0].rfind("_")-2] == "output_scaling_normalize"):
            output_scaling_normalize_models.append(some_list)
    benchmark_models.sort(key = lambda x: x[1])
    output_scaling_models.sort(key = lambda x : x[1])
    output_scaling_normalize_models.sort(key = lambda x : x[1])
    generate_bar_error(benchmark_models[0:5] + output_scaling_models[0:5] + output_scaling_normalize_models[0:5], "Output scaling")
    output_scaling_total_models = []
    for i in range(5):
        output_scaling_total_models.append(benchmark_models[i][0])
        output_scaling_total_models.append(output_scaling_models[i][0])
        output_scaling_total_models.append(output_scaling_normalize_models[i][0])
    model_testing2(output_scaling_total_models, "Output scaling")

    ### Activation functions, normalize
    benchmark_models = []
    tanh_models = []
    mix_models = []
    for some_list in price_mse + imp_mse:
        if (some_list[0][:some_list[0].rfind("_")-2] == "benchmark"):
            benchmark_models.append(some_list)
        elif (some_list[0][:some_list[0].rfind("_")-2] == "tanh"):
            tanh_models.append(some_list)
        elif (some_list[0][:some_list[0].rfind("_")-2] == "mix"):
            mix_models.append(some_list)
    benchmark_models.sort(key = lambda x: x[1])
    tanh_models.sort(key = lambda x : x[1])
    mix_models.sort(key = lambda x : x[1])
    generate_bar_error(benchmark_models[0:5] + tanh_models[0:5] + mix_models[0:5], "Activation functions normalize")
    activiation_function_models = []
    for i in range(5):
        activiation_function_models.append(benchmark_models[i][0])
        activiation_function_models.append(tanh_models[i][0])
        activiation_function_models.append(mix_models[i][0])
    model_testing2(activiation_function_models, "Activation functions normalize")

    ### Activation functions, standardize
    tanh_standard_models = []
    mix_standard_models = []
    standardize_models = []
    for some_list in price_mse + imp_mse:
        if (some_list[0][:some_list[0].rfind("_")-2] == "standardize"):
            standardize_models.append(some_list)
        elif (some_list[0][:some_list[0].rfind("_")-2] == "mix_standardize"):
            mix_standard_models.append(some_list)
        elif (some_list[0][:some_list[0].rfind("_")-2] == "tanh_standardize"):
            tanh_standard_models.append(some_list)
    benchmark_models.sort(key = lambda x: x[1])
    tanh_models.sort(key = lambda x : x[1])
    mix_models.sort(key = lambda x : x[1])
    generate_bar_error(tanh_standard_models[0:5] + mix_standard_models[0:5] + standardize_models[0:5], "Activation functions standardize")
    activiation_function_models = []
    for i in range(5):
        activiation_function_models.append(tanh_standard_models[i][0])
        activiation_function_models.append(mix_standard_models[i][0])
        activiation_function_models.append(standardize_models[i][0])
    model_testing2(activiation_function_models, "Activation functions standardize")

    ### Best activation functions
    all_activation_functions = benchmark_models + tanh_models + mix_models + standardize_models + mix_standard_models + tanh_standard_models
    all_activation_functions.sort(key = lambda x: x[1])
    generate_bar_error(all_activation_functions[0:10], "Activation functions")
    all_activation_functions_list = []
    for i in range(10):
        all_activation_functions_list.append(all_activation_functions[i][0])
    model_testing2(all_activation_functions_list, "Activation functions")

    ### Grid vs sobol
    sobol_models = sobol_models_mse
    grid_models = grid_models_mse
    sobol_models.sort(key = lambda x: x[1])
    grid_models.sort(key = lambda x : x[1])
    generate_bar_error(sobol_models[0:5] + grid_models[0:5], "Sobol vs grid")
    grid_sobol_models = []
    for i in range(5):
        grid_sobol_models.append(sobol_models[i][0])
        grid_sobol_models.append(grid_models[i][0])
    model_testing2(grid_sobol_models, "Sobol vs grid")

    ### Low-high no. data
    benchmark_models = []
    low_data_models = []
    high_data_models = []
    for some_list in combined_mse:
        if (some_list[0][:some_list[0].rfind("_")-2] == "benchmark"):
            benchmark_models.append(some_list)
        elif (some_list[0][:some_list[0].rfind("_")-2] == "low_data"):
           low_data_models.append(some_list)
        elif (some_list[0][:some_list[0].rfind("_")-2] == "high_data"):
            high_data_models.append(some_list)
    benchmark_models.sort(key = lambda x: x[1])
    low_data_models.sort(key = lambda x : x[1])
    high_data_models.sort(key = lambda x : x[1])
    generate_bar_error(benchmark_models[0:5] + low_data_models[0:5] + high_data_models[0:5], "Data size")
    data_no_models = []
    for i in range(5):
        data_no_models.append(benchmark_models[i][0])
        data_no_models.append(low_data_models[i][0])
        data_no_models.append(high_data_models[i][0])
    model_testing2(data_no_models, "Data size")

    ### Standardize vs mat vs single
    standardize_mat_model = glob.glob("Models5/*/standardize_mat*.h5")
    standardize_single_model = glob.glob("Models5/*/standardize_single*.h5")
    mat_single_mse = model_test_set(standardize_mat_model+standardize_single_model, X_test, Y_test, Y_test_price)
    benchmark_models = []
    mat_models = []
    single_models = []
    for some_list in mat_single_mse+imp_models:
        if (some_list[0][:some_list[0].rfind("_")-2] == "standardize"):
            benchmark_models.append(some_list)
        elif (some_list[0][:some_list[0].rfind("_")-2] == "standardize_mat"):
            mat_models.append(some_list)
        elif (some_list[0][:some_list[0].rfind("_")-2] == "standardize_single"):
            single_models.append(some_list)
    benchmark_models.sort(key = lambda x: x[1])
    mat_models.sort(key = lambda x : x[1])
    single_models.sort(key = lambda x : x[1])
    generate_bar_error(benchmark_models[0:5] + mat_models[0:5] + single_models[0:5], "Grid vs smile vs single")
    data_no_models = []
    for i in range(5):
        data_no_models.append(benchmark_models[i][0])
        data_no_models.append(low_data_models[i][0])
        data_no_models.append(high_data_models[i][0])
    model_testing2(data_no_models, "Data size")


    ### Monte Carlo
    mc = [
        #"mc_10", "mc_100", "mc_1000", "mc_10000", 
        "mc_1_price", "mc_10_price", "mc_100_price", "mc_1000_price", "mc_10000_price"#,
        #"mc_1_mat", "mc_10_mat", "mc_100_mat", "mc_1000_mat", "mc_10000_mat",
        #"mc_1_single", "mc_10_single", "mc_100_single", "mc_1000_single", "mc_10000_single"
    ]

    top_mc_models = [
        "Models5/mc_1_mat/mc_1_mat_4_500.h5", "Models5/mc_1_single/mc_1_single_5_1000.h5", "Models5/mc_1_price/mc_1_price_4_100.h5",
        "Models5/mc_10_mat/mc_10_mat_1_1000.h5", "Models5/mc_10_single/mc_10_single_1_50.h5", "Models5/mc_10_price/mc_10_price_5_100.h5", "Models5/mc_10/mc_10_4_500.h5",
        "Models5/mc_100_mat/mc_100_mat_1_1000.h5", "Models5/mc_100_single/mc_100_single_1_100.h5", "Models5/mc_100_price/mc_100_price_4_500.h5", "Models5/mc_100/mc_100_4_1000.h5",
        "Models5/mc_1000_mat/mc_1000_mat_1_500.h5", "Models5/mc_1000_single/mc_1000_single_1_100.h5", "Models5/mc_1000_price/mc_1000_price_4_1000.h5", "Models5/mc_1000/mc_1000_5_100.h5",
        "Models5/mc_10000_mat/mc_10000_mat_2_50.h5", "Models5/mc_10000_single/mc_10000_single_1_50.h5", "Models5/mc_10000_price/mc_10000_price_4_500.h5", "Models5/mc_10000/mc_10000_5_1000.h5"
    ]

    mc_mse2 = model_test_set(top_mc_models, X_test, Y_test, Y_test_price)

    mc_list = []
    for key in mc:
        mc_list.append(glob.glob("Models5/"+key+"/*.h5"))

    mc_list = [item for sublist in mc_list for item in sublist]

    mc_price_mse = model_test_set(mc_list, X_test, Y_test_old, Y_test_price_old)

    mc_mse.sort(key = lambda x: -x[1])
    with open("mc_mse_final.pkl", "wb") as fp:   #Pickling
        pickle.dump(mc_mse, fp)

    with open("mc_mse_final.pkl", "rb") as fp:   # Unpickling
        mc_mse = pickle.load(fp)

    evaluation_first_list = []
    for some_list in mc_mse:
        evaluation_first_list.append(
            [some_list[0][:some_list[0].rfind("_")-2], some_list[0][some_list[0].rfind("_")-1:], '{0:.7f}'.format(round(some_list[1], 7))]
        )
    
    ### Finding best models per group
    evaluation_first_list.sort(key = lambda x: x[0])
    group_by_model = itertools.groupby(evaluation_first_list, key = lambda x: x[0])

    top_first_models_list = []
    for key, group in group_by_model:
        some_list = list(group)
        some_list.sort(key = lambda x: x[2])
        top_first_models_list.append(some_list[0])

    ### Finding best models per setup
    evaluation_setup_list = []
    for some_list in mc_mse:
        evaluation_setup_list.append(
            [some_list[0][:some_list[0].rfind("_")-2], some_list[0][some_list[0].rfind("_")-1:], '{0:.7f}'.format(round(some_list[1], 7))]
        )
    evaluation_setup_list.sort(key = lambda x: x[1])
    group_by_network = itertools.groupby(evaluation_setup_list, key = lambda x: x[1])

    top_first_network_list = []
    for key, group in group_by_network:
        some_list = list(group)
        some_list.sort(key = lambda x: x[2])
        top_first_network_list.append(some_list[0])

    top_first_network_list.sort(key = lambda x: x[1])

    ### Regularization and dropout
    regularization_models = glob.glob("Models5/regularization/*.h5")
    dropout_models = glob.glob("Models5/dropout/*.h5")

    regul_drop_mse = model_test_set(regularization_models+dropout_models, X_test, Y_test, Y_test_price)
    regul_drop_mse.sort(key = lambda x: x[1])

    """
    grid_vs_sobol = glob.glob("Models4/grid_vs_sobol/*.h5")
    #generate_plots(grid_vs_sobol, "grid_vs_sobol")

    output_scaling = glob.glob("Models4/output_scaling/*.h5")
    #generate_plots(output_scaling + base, "output_scaling")

    price_vs_imp = glob.glob("Models4/price_vs_imp/*.h5")
    #generate_plots(price_vs_imp + base, "price_vs_imp")

    activation_functions = glob.glob("Models4/activation_functions/*.h5")
    #generate_plots(activation_functions + base, "activation_functions")

    standard = glob.glob("Models4/standard/*.h5")
    #generate_plots(standard + base, "standard")

    final = glob.glob("Models4/final/*.h5")
    #generate_plots(final, "final")

    final2 = glob.glob("Models4/final2/*.h5")
    #generate_plots(final2, "final2")

    final3 = glob.glob("Models4/final3/*.h5")
    #generate_plots(final3, "final3")

    noise1 = glob.glob("Models4/noise/*.h5")
    noise2 = glob.glob("Models4/noise2/*.h5")

    price_standard = glob.glob("Models4/price_standard/*.h5")
    ### Removing overfitting
    price_standard.remove("Models4/price_standard/price_standard_3_1000.h5")
    price_standard.remove("Models4/price_standard/price_standard_4_1000.h5")
    price_standard.remove("Models4/price_standard/price_standard_5_1000.h5")
    price_standard.remove("Models4/price_standard/price_standard_5_500.h5")
    #generate_plots(price_standard, "price_standard")
    new_data = glob.glob("Models4/new_data/*.h5")
    new_data_mse = model_testing2(new_data, "new_data", easy_case(), hard_case(), option_input())

    new_data_include_zero = glob.glob("Models4/new_data_include_zero/*.h5")
    new_data_include_zero_mse = model_testing2(new_data_include_zero, "net_data_include_zero", easy_case(), hard_case(), option_input())

    mc_models = glob.glob("Models4/mc_*/*.h5", recursive=True)
    mc_models_mse = model_testing2(mc_models, "test_models", easy_case(), hard_case(), option_input())

    ### With test set
    X_test = np.loadtxt("Data/Sobol2_X_test.csv", delimiter = ",")
    Y_test = np.loadtxt("Data/Sobol2_Y_test.csv", delimiter = ",")

    test_list = base + grid_vs_sobol + output_scaling + activation_functions + standard + final + final2 + final3
    mse_list = model_test_set(test_list, X_test, Y_test)
    mse_list.sort(key = lambda x: x[1])
    mse_list[:15]
    generate_bar_error(mse_list[:15], "Top models test set")

    noise_mse = model_test_set(noise1+noise2, X_test, Y_test)
    noise_mse.sort(key = lambda x: x[1])

    ### To look at the best models
    top_models = []
    for i in range(15):
        top_models.append(mse_list[i][0])

    ### Generate surfaces
    generate_plots(["Models4/" + s for s in top_models], "Top models")

    ### Testing simulation models
    mc_1 = glob.glob("Models4/mc_1/*.h5")
    mc_10 = glob.glob("Models4/mc_10/*.h5")
    mc_100 = glob.glob("Models4/mc_100/*.h5")
    mc_1000 = glob.glob("Models4/mc_1000/*.h5")
    mc_10000 = glob.glob("Models4/mc_10000/*.h5")
    mc_1_price = glob.glob("Models4/mc_1/price/*.h5")
    mc_10_price = glob.glob("Models4/mc_10/price/*.h5")
    mc_100_price = glob.glob("Models4/mc_100/price/*.h5")
    mc_1000_price = glob.glob("Models4/mc_1000/price/*.h5")
    mc_10000_price = glob.glob("Models4/mc_10000/price/*.h5")

    mc_input = np.loadtxt("Data/MC/HestonMC_input.csv", delimiter=",")

    al_imp = np.loadtxt("Data/hestonSobolGridImpVol2_200000.csv", delimiter=",")
    al_price = np.loadtxt("Data/hestonSobolGridPrice2_200000.csv", delimiter=",")

    test_index = np.loadtxt("Data/MC/test_index.csv", delimiter=",").astype(int)

    y_test_mc = al_imp[test_index, :]
    y_test_price_mc = al_price[test_index, :]

    x_test_mc = mc_input[test_index, :]

    imp_monte_carlo = mc_1 + mc_10 + mc_100 + mc_1000 + mc_10000
    price_monte_carlo = mc_1_price + mc_10_price + mc_100_price + mc_1000_price + mc_10000_price

    mc_mse = monte_carlo_testing(imp_monte_carlo, x_test_mc, y_test_mc)
    mc_mse.sort(key = lambda x: x[1])

    mc_mse_price = monte_carlo_testing(price_monte_carlo, x_test_mc, y_test_price_mc)
    mc_mse_price.sort(key = lambda x: x[1])
    """


"""
norm_folder = "Models4/norms/"

### Grid data, Sobol
input_pre = np.loadtxt("Data/hestonSobolGridInput2_compare2_200000.csv", delimiter = ",")
output_pre = np.loadtxt("Data/hestonSobolGridImpVol2_compare2_200000.csv", delimiter = ",")
output_pre_price = np.loadtxt("Data/hestonSobolGridPrice2_compare2_200000.csv", delimiter = ",")

sobol_filter = np.all(output_pre != 0, axis = 1)
sobol_input = input_pre[sobol_filter, :]
sobol_output = output_pre[sobol_filter, :]

sobol_filter_price = np.all(output_pre != 0, axis = 1)
sobol_input_price = input_pre[sobol_filter_price, :]
sobol_output_price = output_pre_price[sobol_filter_price, :]

X_train, X_test, Y_train, Y_test = train_test_split(sobol_input, sobol_output, test_size=0.3, random_state=42)
X_train_price, X_test_price, Y_train_price, Y_test_price = train_test_split(sobol_input_price, sobol_output_price, test_size=0.3, random_state=42)

norm_feature_standard = MinMaxScaler()
standard_feature_standard = StandardScaler()
standard_label_standard = StandardScaler()
price_label_standard = StandardScaler()

norm_feature_standard.fit(X_train)
standard_feature_standard.fit(X_train)
standard_label_standard.fit(Y_train)
price_label_standard.fit(Y_train_price)

joblib.dump(norm_feature_standard, norm_folder+"norm_features.pkl")
joblib.dump(standard_feature_standard, norm_folder+"standard_features.pkl")
joblib.dump(standard_label_standard, norm_folder+"norm_labels.pkl")
joblib.dump(price_label_standard, norm_folder+"norm_labels_price.pkl")

norm_feature_price = MinMaxScaler()

norm_feature_price.fit(X_train_price)

joblib.dump(norm_feature_price, norm_folder+"norm_feature_price.pkl")

np.savetxt("Data/Sobol2_X_train.csv", X_train, delimiter = ",")
np.savetxt("Data/Sobol2_X_test.csv", X_test, delimiter = ",")
np.savetxt("Data/Sobol2_Y_train.csv", Y_train, delimiter = ",")
np.savetxt("Data/Sobol2_Y_test.csv", Y_test, delimiter = ",")

np.savetxt("Data/Sobol2_X_train_price.csv", X_train_price, delimiter = ",")
np.savetxt("Data/Sobol2_X_test_price.csv", X_test_price, delimiter = ",")
np.savetxt("Data/Sobol2_Y_train_price.csv", Y_train_price, delimiter = ",")
np.savetxt("Data/Sobol2_Y_test_price.csv", Y_test_price, delimiter = ",")

### Grid data, grid
input_pre_grid = np.loadtxt("Data/hestonGridInput2_wide.csv", delimiter = ",")
output_pre_grid = np.loadtxt("Data/hestonGridImpVol2_wide.csv", delimiter = ",")

sobol_grid_filter = np.all(output_pre_grid != 0, axis = 1)
sobol_grid_input = input_pre_grid[sobol_grid_filter, :]
sobol_grid_output = output_pre_grid[sobol_grid_filter, :]

X_train_grid, X_test_grid, Y_train_grid, Y_test_grid = train_test_split(sobol_grid_input, sobol_grid_output, test_size=0.3, random_state=42)

norm_feature_grid = MinMaxScaler()

norm_feature_grid.fit(X_train_grid)

joblib.dump(norm_feature_grid, norm_folder+"norm_feature_grid.pkl")

np.savetxt("Data/Sobol2_X_train_grid.csv", X_train_grid, delimiter = ",")
np.savetxt("Data/Sobol2_X_test_grid.csv", X_test_grid, delimiter = ",")
np.savetxt("Data/Sobol2_Y_train_grid.csv", Y_train_grid, delimiter = ",")
np.savetxt("Data/Sobol2_Y_test_grid.csv", Y_test_grid, delimiter = ",")

### Big sobol for comparison
input_wide_pre = np.loadtxt("Data/hestonSobolGridInput2_compare_279936.csv", delimiter = ",")
output_wide_pre = np.loadtxt("Data/hestonSobolGridImpVol2_compare_279936.csv", delimiter = ",")

sobol_wide_filter = np.all(output_wide_pre != 0, axis = 1)
sobol_wide_input = input_wide_pre[sobol_wide_filter, :]
sobol_wide_output = output_wide_pre[sobol_wide_filter, :]

X_train_wide, X_test_wide, Y_train_wide, Y_test_wide = train_test_split(sobol_wide_input, sobol_wide_output, test_size=0.3, random_state=42)

norm_feature_wide = MinMaxScaler()

norm_feature_wide.fit(X_train_wide)

joblib.dump(norm_feature_wide, norm_folder+"norm_feature_wide.pkl")

np.savetxt("Data/Sobol2_X_train_wide.csv", X_train_wide, delimiter = ",")
np.savetxt("Data/Sobol2_X_test_wide.csv", X_test_wide, delimiter = ",")
np.savetxt("Data/Sobol2_Y_train_wide.csv", Y_train_wide, delimiter = ",")
np.savetxt("Data/Sobol2_Y_test_wide.csv", Y_test_wide, delimiter = ",")

### Single data
option_input = dg.option_input_generator()

total_comb = np.shape(input_pre)[0] * np.shape(option_input)[0]
total_cols = np.shape(input_pre)[1] + np.shape(option_input)[1]
total_options = np.shape(option_input)[0]
single_input = np.empty((total_comb, total_cols))
single_output = np.empty((total_comb, 1))
for i in range(np.shape(input_pre)[0]):
    for j in range(total_options):
        single_input[i*total_options+j, 0:np.shape(input_pre)[1]] = input_pre[i]
        single_input[i*total_options+j, (np.shape(input_pre)[1]) : total_cols] = option_input[j]
        single_output[i*total_options+j] = output_pre[i, j]
    
single_output = single_output.flatten()
single_output = np.reshape(single_output, (-1, 1))

sobol_single_filter = np.all(single_output != 0, axis = 1)
single_input = single_input[sobol_single_filter, :]
single_output = single_output[sobol_single_filter, :]

X_train_single, X_test_single, Y_train_single, Y_test_single = train_test_split(single_input, single_output, test_size=0.3, random_state=42)

norm_feature_single = MinMaxScaler()

norm_feature_single.fit(X_train_single)

joblib.dump(norm_feature_single, norm_folder+"norm_feature_single.pkl")

np.savetxt("Data/Sobol2_X_train_single.csv", X_train_single, delimiter = ",")
np.savetxt("Data/Sobol2_X_test_single.csv", X_test_single, delimiter = ",")
np.savetxt("Data/Sobol2_Y_train_single.csv", Y_train_single, delimiter = ",")
np.savetxt("Data/Sobol2_Y_test_single.csv", Y_test_single, delimiter = ",")

"""