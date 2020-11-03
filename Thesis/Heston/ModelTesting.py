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
vol = 0.1

# kappa
kappa = 0.5

# theta
theta = 0.1

# epsilon
epsilon = 1

# rho
rho = -0.5

# rate
rate = 0.05

# Which options to test
# Maturity
maturity = np.linspace(start = 0.01, stop = 2, num = 5)

# strike
no_strikes = 5
strike = np.linspace(start = 75, stop = 125, num = 5)

option_input = np.array(list(itertools.product(maturity, strike))) # different option combinations
some_option_list = np.array([])
for option in option_input:
    some_option_list = np.append(some_option_list, vo.EUCall(option[0], option[1]))

# Dataframe for NN with multiple outputs
test_input = np.array((forward, vol, kappa, theta, epsilon, rho, rate))
test_input = np.reshape(test_input, (1, 7))
model_class = hm.HestonClass(forward, vol, kappa, theta, epsilon, rho, rate)

# Generating benchmark data
benchmark = dg.calc_imp_vol(test_input[0], some_option_list)[1]

file_list = glob.glob("Models/Heston/*.h5")
models = []
ending = []
i = 0
for some_file in file_list:
    models.append(some_file[:-5])
    ending.append(some_file[-4:-3])

#fig, axs = plt.subplots(5, 2)
fig = plt.figure() 
grid_imp_ax = fig.add_subplot(421, projection='3d')
grid_imp_ax2 = fig.add_subplot(422)

grid_price_ax = fig.add_subplot(423, projection='3d')
grid_price_ax2 = fig.add_subplot(424)

sobol_grid_imp_ax = fig.add_subplot(425, projection='3d')
sobol_grid_imp_ax2 = fig.add_subplot(426)

sobol_grid_price_ax = fig.add_subplot(427, projection='3d')
sobol_grid_price_ax2 = fig.add_subplot(428)

# Plotting
no_models = len(file_list)
color=iter(plt.cm.rainbow(np.linspace(0,1,no_models)))
x = option_input[:,0]
y = option_input[:,1]
mse_list = []
grid_imp_mse = []
grid_price_mse = []
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
    j += 1

mse_list.sort(key = lambda x: x[1]) # sort by error
grid_imp_mse.sort(key = lambda x: x[1])
grid_price_mse.sort(key = lambda x: x[1])
sobol_grid_imp_mse.sort(key = lambda x: x[1])
sobol_grid_price_mse.sort(key = lambda x: x[1])
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



"""
fig.subplots_adjust(top=0.95, left=0.1, right=0.95, bottom=0.3)
handles, labels = axs[4,1].get_legend_handles_labels() 
fig.legend(handles, labels, loc="lower center", ncol = 4, fontsize=5)
plt.savefig("HestonModelTestPlot.jpeg")
"""
plt.savefig("HestonModelsCompare.jpeg")
plt.show()