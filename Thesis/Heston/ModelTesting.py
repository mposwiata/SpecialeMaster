import numpy as np
import itertools
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
import glob
from sklearn.metrics import mean_squared_error as mse

from Thesis.Heston import AndersenLake as al, HestonModel as hm, DataGeneration as dg
from Thesis.misc import VanillaOptions as vo

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

# Which options to test
# Maturity
maturity = np.linspace(start = 0.01, stop = 2, num = 5)

# strike
no_strikes = 5
strike = np.linspace(start = 75, stop = 125, num = 5)

option_input = np.array(list(itertools.product(maturity, strike))) # different option combinations

# Dataframe for NN with multiple outputs
test_input = np.array((forward, vol, kappa, theta, epsilon, rho, rate))
test_input = np.reshape(test_input, (1, 7))

def find_nth_back(input_string : str, keyword : str, n : int):
    start = input_string.rfind("_")
    while start >= 0 and n > 1:
        start = input_string.rfind("_", 0, start)
        n -= 1
    return start

models = glob.glob("Models2/Heston*/*.h5")

group_by_list = []
for some_model in models:
    group_by_list.append(some_model[:find_nth_back(some_model, "_", 3)])

plot_model_dict = {key : [value for value in models if key in value] for key in set(group_by_list)}

def model_testing_plot(model_list : list, plot_group : str, some_input : np.ndarray, option : np.ndarray):
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

        if (model_string.find("Single") != -1): # single output
            predictions = np.empty(np.shape(option_input)[0])
            for i in range(np.shape(option_input)[0]):
                test_single_input = np.concatenate((some_input, option_input[i]), axis=None)
                test_single_input = np.reshape(test_single_input, (1, -1))
                if normal_out:
                    predictions[i] = norm_labels.inverse_transform(model.predict(norm_feature.transform(test_single_input)))
                else:
                    predictions[i] = model.predict(norm_feature.transform(test_single_input))
        else: # we have a grid
            if normal_out:
                predictions = norm_labels.inverse_transform(model.predict(norm_feature.transform(test_input)))[0]
            else:
                predictions = model.predict(norm_feature.transform(some_input))[0]

        # if prices, calc imp vol
        if (model_string.find("Price") != -1 or model_string.find("price") != -1 ):
            imp_vol_predictions = np.empty(np.shape(predictions))
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
    
    imp_ax.plot_trisurf(x, y, benchmark, color = "black", alpha = 0.5)

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
    fig.suptitle(plot_group[plot_group.find("/")+1:].replace("/", "_"))
    plt.savefig("Plots2/"+plot_group[7:].replace("/", "_")+".png")
    plt.close()
    return mse_list

mse_bar_list = []
for model_group in plot_model_dict:
    mse_bar_list.append(model_testing_plot(plot_model_dict[model_group], model_group, test_input, option_input))

low_error_mse = []
med_error_mse = []
high_error_mse = []
for model_list in mse_bar_list:
    for some_model in model_list:
        if some_model[1] < 0.001:
            low_error_mse.append(some_model)
        elif some_model[1] < 0.5:
            med_error_mse.append(some_model)
        else:
            high_error_mse.append(some_model)

def generate_bar_error(error_list : list, name : str):
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

generate_bar_error(low_error_mse, "low_error")
generate_bar_error(med_error_mse, "med_error")
generate_bar_error(high_error_mse, "high_error")

"""
#fig, axs = plt.subplots(5, 2)
fig = plt.figure() 
grid_imp_ax = fig.add_subplot(621, projection='3d')
grid_imp_ax2 = fig.add_subplot(622)

grid_price_ax = fig.add_subplot(623, projection='3d')
grid_price_ax2 = fig.add_subplot(624)

sobol_grid_imp_ax = fig.add_subplot(625, projection='3d')
sobol_grid_imp_ax2 = fig.add_subplot(626)

sobol_grid_price_ax = fig.add_subplot(627, projection='3d')
sobol_grid_price_ax2 = fig.add_subplot(628)

single_imp_ax = fig.add_subplot(629, projection = '3d')
single_imp_ax2 = fig.add_subplot(6,2,10)

single_price_ax = fig.add_subplot(6, 2, 11, projection = '3d')
single_price_ax2 = fig.add_subplot(6,2,12)

# Plotting
no_models = len(file_list)
color=iter(plt.cm.rainbow(np.linspace(0,1,no_models)))
x = option_input[:,0]
y = option_input[:,1]
mse_list = []
grid_imp_mse = []
single_imp_mse = []
grid_price_mse = []
single_price_mse = []
sobol_grid_imp_mse = []
sobol_grid_price_mse = []
j = 0
for model_string in models:
    if (model_string.find("Sobol") != -1):
        name = "Sobol_"
    else:
        name = "Normal_"

    if (model_string.find("Single") != -1): # single output
        name = name + "single_"
    else:
        name = name + "grid_"

    if (model_string.find("Price") != -1):
        name = name + "price_"
    else:
        name = name + "imp_vol_"
    
    model = load_model(model_string+"_"+ending[j]+".h5")
    norm_feature = joblib.load(model_string+"_norm_features_"+ending[j]+".pkl")
    norm_labels = joblib.load(model_string+"_norm_labels_"+ending[j]+".pkl")

    if (model_string.find("Single") != -1): # single output
        predictions = np.empty(np.shape(option_input)[0])
        for i in range(np.shape(option_input)[0]):
            test_single_input = np.concatenate((test_input, option_input[i]), axis=None)
            test_single_input = np.reshape(test_single_input, (1, -1))
            predictions[i] = norm_labels.inverse_transform(model.predict(norm_feature.transform(test_single_input)))
    else: # we have a grid
        predictions = norm_labels.inverse_transform(model.predict(norm_feature.transform(test_input)))[0]            

    # if prices, calc imp vol
    if (model_string.find("Price") != -1):
        imp_vol_predictions = np.empty(np.shape(predictions))
        for i in range(np.shape(predictions)[0]):
            imp_vol_predictions[i] = model_class.impVol(predictions[i], some_option_list[i])
            predictions = imp_vol_predictions

    if (model_string.find("Sobol") != -1):
        name = name + model_string[model_string.rfind("_")-3:]
    else: 
        name = name + model_string[model_string.rfind("_")-1:]

    mse_list.append((name, mse(predictions, benchmark)))
    c = next(color)

    z = predictions
    if (mse(predictions, benchmark) < 0.5):
        if (model_string.find("Single") == -1):
            if (model_string.find("Sobol") != -1):
                if (model_string.find("Price") != -1):
                    sobol_grid_price_ax.plot_trisurf(x, y, z, alpha = 0.5, label = name)
                    sobol_grid_price_mse.append((name, mse(predictions, benchmark)))
                else:
                    sobol_grid_imp_ax.plot_trisurf(x, y, z, alpha = 0.5, label = name)
                    sobol_grid_imp_mse.append((name, mse(predictions, benchmark)))
            else:
                if (model_string.find("Price") != -1):
                    grid_price_ax.plot_trisurf(x, y, z, alpha = 0.5, label = name)
                    grid_price_mse.append((name, mse(predictions, benchmark)))
                else:
                    grid_imp_ax.plot_trisurf(x, y, z, alpha = 0.5, label = name)
                    grid_imp_mse.append((name, mse(predictions, benchmark)))
        else:
            if (model_string.find("Price") != -1):
                single_price_ax.plot_trisurf(x, y, z, alpha = 0.5, label = name)
                single_price_mse.append((name, mse(predictions, benchmark)))
            else:
                single_imp_ax.plot_trisurf(x, y, z, alpha = 0.5, label = name)
                single_imp_mse.append((name, mse(predictions, benchmark)))

    j += 1

mse_list.sort(key = lambda x: x[1]) # sort by error
grid_imp_mse.sort(key = lambda x: x[1])
grid_price_mse.sort(key = lambda x: x[1])
sobol_grid_imp_mse.sort(key = lambda x: x[1])
sobol_grid_price_mse.sort(key = lambda x: x[1])
single_imp_mse.sort(key = lambda x: x[1])
single_price_mse.sort(key = lambda x: x[1])
filtered_mse = list(filter(lambda x: (x[1] < 0.01), mse_list))

# grid imp mse
labels, values = zip(*grid_imp_mse)
y_pos = np.arange(len(labels))
grid_imp_ax2.barh(y_pos, values)
grid_imp_ax2.set_yticks(y_pos)
grid_imp_ax2.set_yticklabels(labels)
grid_imp_ax2.invert_yaxis()
grid_imp_ax2.tick_params(axis = "y", labelsize=3)
grid_imp_ax.tick_params(labelsize=5)

# grid price mse
labels, values = zip(*grid_price_mse)
y_pos = np.arange(len(labels))
grid_price_ax2.barh(y_pos, values)
grid_price_ax2.set_yticks(y_pos)
grid_price_ax2.set_yticklabels(labels)
grid_price_ax2.invert_yaxis()
grid_price_ax2.tick_params(axis = "y", labelsize=3)
grid_price_ax.tick_params(labelsize=5)

# sobol grid imp mse
labels, values = zip(*sobol_grid_imp_mse)
y_pos = np.arange(len(labels))
sobol_grid_imp_ax2.barh(y_pos, values)
sobol_grid_imp_ax2.set_yticks(y_pos)
sobol_grid_imp_ax2.set_yticklabels(labels)
sobol_grid_imp_ax2.invert_yaxis()
sobol_grid_imp_ax2.tick_params(axis = "y", labelsize=3)
sobol_grid_imp_ax.tick_params(labelsize=5)

# sobol grid price mse
labels, values = zip(*sobol_grid_price_mse)
y_pos = np.arange(len(labels))
sobol_grid_price_ax2.barh(y_pos, values)
sobol_grid_price_ax2.set_yticks(y_pos)
sobol_grid_price_ax2.set_yticklabels(labels)
sobol_grid_price_ax2.invert_yaxis()
sobol_grid_price_ax2.tick_params(axis = "y", labelsize=3)
sobol_grid_price_ax.tick_params(labelsize = 5)

# single price mse
labels, values = zip(*single_price_mse)
y_pos = np.arange(len(labels))
single_price_ax2.barh(y_pos, values)
single_price_ax2.set_yticks(y_pos)
single_price_ax2.set_yticklabels(labels)
single_price_ax2.invert_yaxis()
single_price_ax2.tick_params(axis = "y", labelsize=3)
single_price_ax2.tick_params(labelsize = 5)

# single imp mse
labels, values = zip(*single_imp_mse)
y_pos = np.arange(len(labels))
single_imp_ax2.barh(y_pos, values)
single_imp_ax2.set_yticks(y_pos)
single_imp_ax2.set_yticklabels(labels)
single_imp_ax2.invert_yaxis()
single_imp_ax2.tick_params(axis = "y", labelsize=3)
single_imp_ax2.tick_params(labelsize = 5)

fig.subplots_adjust(top=0.95, left=0.1, right=0.95, bottom=0.3)
handles, labels = axs[4,1].get_legend_handles_labels() 
fig.legend(handles, labels, loc="lower center", ncol = 4, fontsize=5)
plt.savefig("HestonModelTestPlot.jpeg")
#plt.savefig("HestonModelsCompare.jpeg")
plt.show()
"""