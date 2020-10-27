import numpy as np
import itertools
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model

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
benchmark = dg.calcImpVol(test_input[0], some_option_list)[1]

grid_imp_vol_list = [
    "HestonGridImpVolAll_100", "HestonGridImpVolAll_1000", 
    "HestonGridImpVolFilter_100", "HestonGridImpVolFilter_1000"
]

grid_price_list = [
    "HestonGridPriceAll_100", "HestonGridPriceAll_1000",
    "HestonGridPriceFilter_100"#, 
    #"HestonGridPriceFilter_1000"
]

single_imp_vol_list = [
    "HestonSingleImpVolAll_100", "HestonSingleImpVolAll_1000",
    "HestonSingleImpVolFilter_100", "HestonSingleImpVolFilter_1000"
]

single_price_list = [
    #"HestonSinglePriceAll_100", 
    #"HestonSinglePriceAll_1000",
    "HestonSinglePriceFilter_100"#, 
    #"HestonSinglePriceFilter_1000"
]
fig, axs = plt.subplots(5, 2)
# Grid Imp Vol Plots
color=iter(plt.cm.rainbow(np.linspace(0,1,8)))
for model_string in grid_imp_vol_list:
    model = load_model("Models/Heston/"+model_string+"_1.h5")
    norm_feature = joblib.load("Models/Heston/"+model_string+"_norm_features_1.pkl")
    norm_labels = joblib.load("Models/Heston/"+model_string+"_norm_labels_1.pkl")
    predictions = norm_labels.inverse_transform(model.predict(norm_feature.transform(test_input)))
    c = next(color)
    for i in range(5):
        axs[i, 0].plot(option_input[0 : no_strikes, 1], predictions[0, no_strikes * i : no_strikes * (i + 1)], color = c, alpha = 0.5, label = model_string)
        axs[i, 1].plot(option_input[0 : no_strikes, 1], predictions[0, no_strikes * i : no_strikes * (i + 1)] - benchmark[no_strikes * i : no_strikes * (i + 1)], color = c, alpha = 0.5, label = model_string)

# Grid Price Plots
for model_string in grid_price_list:
    model = load_model("Models/Heston/"+model_string+"_1.h5")
    norm_feature = joblib.load("Models/Heston/"+model_string+"_norm_features_1.pkl")
    norm_labels = joblib.load("Models/Heston/"+model_string+"_norm_labels_1.pkl")
    predictions = norm_labels.inverse_transform(model.predict(norm_feature.transform(test_input)))
    imp_vol_predictions = np.empty(np.shape(predictions)[1])
    for i in range(np.shape(predictions)[1]):
        imp_vol_predictions[i] = model_class.impVol(predictions[0, i], some_option_list[i])
    c = next(color)
    for i in range(5):
        axs[i, 0].plot(option_input[0 : no_strikes, 1], imp_vol_predictions[no_strikes * i : no_strikes * (i + 1)], color = c, alpha = 0.5, label = model_string)
        #axs[i, 0].plot(option_input[0 : no_strikes, 1], benchmark[no_strikes * i : no_strikes * (i + 1)], color = "black", alpha = 0.7, label = "benchmark")
        axs[i, 1].plot(option_input[0 : no_strikes, 1], imp_vol_predictions[no_strikes * i : no_strikes * (i + 1)] - benchmark[no_strikes * i : no_strikes * (i + 1)], color = c, alpha = 0.5, label = model_string)

"""
# Single Imp Vol Plots
for model_string in single_imp_vol_list:
    model = load_model("Models/Heston/"+model_string+"_1.h5")
    norm_feature = joblib.load("Models/Heston/"+model_string+"_norm_features_1.pkl")
    norm_labels = joblib.load("Models/Heston/"+model_string+"_norm_labels_1.pkl")
    c = next(color)
    test_length = np.shape(option_input)[0]
    predictions = np.empty(test_length)
    for i in range(test_length):
        test_data = np.concatenate((test_input, option_input[i]), axis=None)
        test_data = np.reshape(test_data, (1, -1))
        predictions[i] = norm_labels.inverse_transform(model.predict(norm_feature.transform(test_data)))

    for i in range(5):
        axs[i, 0].plot(option_input[0 : no_strikes, 1], predictions[no_strikes * i : no_strikes * (i + 1)], color = c, alpha = 0.5, label = model_string)
        axs[i, 1].plot(option_input[0 : no_strikes, 1], predictions[no_strikes * i : no_strikes * (i + 1)] - benchmark[no_strikes * i : no_strikes * (i + 1)], color = c, alpha = 0.5, label = model_string)
"""

# Single Price Plots
for model_string in single_price_list:
    model = load_model("Models/Heston/"+model_string+"_1.h5")
    norm_feature = joblib.load("Models/Heston/"+model_string+"_norm_features_1.pkl")
    norm_labels = joblib.load("Models/Heston/"+model_string+"_norm_labels_1.pkl")
    c = next(color)
    test_length = np.shape(option_input)[0]
    predictions = np.empty(test_length)
    for i in range(test_length):
        test_data = np.concatenate((test_input, option_input[i]), axis=None)
        test_data = np.reshape(test_data, (1, -1))
        price = norm_labels.inverse_transform(model.predict(norm_feature.transform(test_data)))
        predictions[i] = model_class.impVol(price[0,0], some_option_list[i])

    for i in range(5):
        axs[i, 0].plot(option_input[0 : no_strikes, 1], predictions[no_strikes * i : no_strikes * (i + 1)], color = c, alpha = 0.5, label = model_string)
        axs[i, 1].plot(option_input[0 : no_strikes, 1], predictions[no_strikes * i : no_strikes * (i + 1)] - benchmark[no_strikes * i : no_strikes * (i + 1)], color = c, alpha = 0.5, label = model_string)

fig.subplots_adjust(top=0.95, left=0.1, right=0.95, bottom=0.3)
handles, labels = axs[4,1].get_legend_handles_labels() 
fig.legend(handles, labels, loc="lower center", ncol = 2, fontsize=8)
plt.savefig("HestonModelTestPlot.jpeg")
plt.show()