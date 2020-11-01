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

file_list = glob.glob("Models/Heston1/*.h5")
models = []
ending = []
i = 0
for some_file in file_list:
    models.append(some_file[:-5])
    ending.append(some_file[-4:-3])

#fig, axs = plt.subplots(5, 2)
fig = plt.figure() 
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
# Plotting
no_models = len(file_list)
color=iter(plt.cm.rainbow(np.linspace(0,1,no_models)))
x = option_input[:,0]
y = option_input[:,1]
mse_list = []
j = 0
for model_string in models:
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
        for i in range(np.shape(predictions)):
            imp_vol_predictions[i] = model_class.impVol(predictions[i], some_option_list[i])
            predictions = imp_vol_predictions
    
    mse_list.append((model_string, mse(predictions, benchmark)))
    c = next(color)

    z = predictions
    ax.plot_trisurf(x, y, z, alpha = 0.5, label = model_string[14:])
    
    """
    for i in range(5):
        axs[i, 0].plot(option_input[0 : no_strikes, 1], predictions[0, no_strikes * i : no_strikes * (i + 1)], color = c, alpha = 0.5, label = model_string[14:])
        axs[i, 1].plot(option_input[0 : no_strikes, 1], predictions[0, no_strikes * i : no_strikes * (i + 1)] - benchmark[no_strikes * i : no_strikes * (i + 1)], color = c, alpha = 0.5, label = model_string[14:])
        axs[i, 0].plot(option_input[0 : no_strikes, 1], benchmark[no_strikes * i : no_strikes * (i + 1)], color = "black", alpha = 0.7, label = "benchmark")
    """

    j += 1

mse_list.sort(key = lambda x: x[1]) # sort by error
"""
fig.subplots_adjust(top=0.95, left=0.1, right=0.95, bottom=0.3)
handles, labels = axs[4,1].get_legend_handles_labels() 
fig.legend(handles, labels, loc="lower center", ncol = 4, fontsize=5)
plt.savefig("HestonModelTestPlot.jpeg")
"""
plt.show()