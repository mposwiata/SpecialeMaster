import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras import backend as k
from multiprocess import Pool, cpu_count, Process
import joblib
import sys
import os
sys.path.append(os.getcwd()) # added for calc server support

from Thesis import NeuralNetworkGenerator as nng
from sklearn.model_selection import train_test_split
from Thesis.Heston import NNModelGenerator as mg
from Thesis.Heston import DataGeneration as dg

model_sobol_input_100000 = np.loadtxt("Data/hestonSobolGridInput_100000.csv", delimiter = ",")
model_sobol_input_200000 = np.loadtxt("Data/hestonSobolGridInput_200000.csv", delimiter = ",")
model_sobol_input_312500 = np.loadtxt("Data/hestonSobolGridInput_312500.csv", delimiter = ",")
model_sobol2_input_100000= np.loadtxt("Data/hestonSobolGridInput2_100000.csv", delimiter = ",")
model_sobol2_input_200000= np.loadtxt("Data/hestonSobolGridInput2_200000.csv", delimiter = ",")
model_sobol2_input_312500 = np.loadtxt("Data/hestonSobolGridInput2_312500.csv", delimiter = ",")

grid_price_output = np.loadtxt("Data/hestonGridPrice.csv", delimiter = ",")
grid_imp_vol_output = np.loadtxt("Data/hestonGridImpVol.csv", delimiter = ",")
sobol_price_100000 = np.loadtxt("Data/hestonSobolGridPrice_100000.csv", delimiter = ",")
sobol_price_200000 = np.loadtxt("Data/hestonSobolGridPrice_200000.csv", delimiter = ",")
sobol_price_312500 = np.loadtxt("Data/hestonSobolGridPrice_312500.csv", delimiter = ",")
sobol2_price_100000 = np.loadtxt("Data/hestonSobolGridPrice2_100000.csv", delimiter = ",")
sobol2_price_200000 = np.loadtxt("Data/hestonSobolGridPrice2_200000.csv", delimiter = ",")
sobol2_price_312500 = np.loadtxt("Data/hestonSobolGridPrice2_312500.csv", delimiter = ",")
sobol_imp_vol_100000 = np.loadtxt("Data/hestonSobolGridImpVol_100000.csv", delimiter = ",")
sobol2_imp_vol_100000 = np.loadtxt("Data/hestonSobolGridImpVol2_100000.csv", delimiter = ",")
sobol_imp_vol_200000 = np.loadtxt("Data/hestonSobolGridImpVol_200000.csv", delimiter = ",")
sobol2_imp_vol_200000 = np.loadtxt("Data/hestonSobolGridImpVol2_200000.csv", delimiter = ",")
sobol_imp_vol_312500 = np.loadtxt("Data/hestonSobolGridImpVol_312500.csv", delimiter = ",")
sobol2_imp_vol_312500 = np.loadtxt("Data/hestonSobolGridImpVol2_312500.csv", delimiter = ",")

model_input = model_sobol2_input_312500
option_input = dg.option_input_generator()

total_comb = np.shape(model_input)[0] * np.shape(option_input)[0]
total_cols = np.shape(model_input)[1] + np.shape(option_input)[1]
total_options = np.shape(option_input)[0]
sobol2_single_input = np.empty((total_comb, total_cols))
sobol2_single_price_output = np.empty((total_comb, 1))
sobol2_single_imp_vol_output = np.empty((total_comb, 1))
for i in range(np.shape(model_input)[0]):
    for j in range(total_options):
        sobol2_single_input[i*total_options+j, 0:np.shape(model_input)[1]] = model_input[i]
        sobol2_single_input[i*total_options+j, (np.shape(model_input)[1]) : total_cols] = option_input[j]
        sobol2_single_price_output[i*total_options+j] = sobol2_price_312500[i, j]
        sobol2_single_imp_vol_output[i*total_options+j] = sobol2_imp_vol_312500[i, j]

sobol2_single_price_output = sobol2_single_price_output.flatten()
sobol2_single_imp_vol_output = sobol2_single_imp_vol_output.flatten()
model_input = dg.model_input_generator() #faster than reading file


# Grid filtering, rows with 0 in
filtered_price_grid = np.all(grid_price_output != 0, axis = 1)
filtered_grid_model_price = model_input[filtered_price_grid, :]
filtered_grid_price = grid_price_output[filtered_price_grid, :]

filtered_imp_vol_grid = np.all(grid_imp_vol_output != 0, axis = 1)
filtered_grid_model_imp_vol = model_input[filtered_imp_vol_grid, :]
filtered_grid_imp_vol = grid_imp_vol_output[filtered_imp_vol_grid, :]

sobol_imp_vol_100_filter = np.all(sobol_imp_vol_100000 != 0, axis = 1)
sobol_imp_vol_100_input = model_sobol_input_100000[sobol_imp_vol_100_filter, :]
sobol_imp_vol_100_output = sobol_imp_vol_100000[sobol_imp_vol_100_filter, :]

sobol_imp_vol_200_filter = np.all(sobol_imp_vol_200000 != 0, axis = 1)
sobol_imp_vol_200_input = model_sobol_input_200000[sobol_imp_vol_200_filter, :]
sobol_imp_vol_200_output = sobol_imp_vol_200000[sobol_imp_vol_200_filter, :]

sobol_imp_vol_312_filter = np.all(sobol_imp_vol_312500 != 0, axis = 1)
sobol_imp_vol_312_input = model_sobol_input_312500[sobol_imp_vol_312_filter, :]
sobol_imp_vol_312_output = sobol_imp_vol_312500[sobol_imp_vol_312_filter, :]

sobol2_imp_vol_100_filter = np.all(sobol2_imp_vol_100000 != 0, axis = 1)
sobol2_imp_vol_100_input = model_sobol2_input_100000[sobol2_imp_vol_100_filter, :]
sobol2_imp_vol_100_output = sobol2_imp_vol_100000[sobol2_imp_vol_100_filter, :]

sobol2_imp_vol_200_filter = np.all(sobol2_imp_vol_200000 != 0, axis = 1)
sobol2_imp_vol_200_input = model_sobol2_input_200000[sobol2_imp_vol_200_filter, :]
sobol2_imp_vol_200_output = sobol2_imp_vol_200000[sobol2_imp_vol_200_filter, :]

sobol2_imp_vol_312_filter = np.all(sobol2_imp_vol_312500 != 0, axis = 1)
sobol2_imp_vol_312_input = model_sobol2_input_312500[sobol2_imp_vol_312_filter, :]
sobol2_imp_vol_312_output = sobol2_imp_vol_312500[sobol2_imp_vol_312_filter, :]

sobol_price_100_filter = np.all(sobol_price_100000 != 0, axis = 1)
sobol_price_100_input = model_sobol_input_100000[sobol_price_100_filter, :]
sobol_price_100_output = sobol_price_100000[sobol_price_100_filter, :]

sobol_price_200_filter = np.all(sobol_price_200000 != 0, axis = 1)
sobol_price_200_input = model_sobol_input_200000[sobol_price_200_filter, :]
sobol_price_200_output = sobol_price_200000[sobol_price_200_filter, :]

sobol_price_312_filter = np.all(sobol_price_312500 != 0, axis = 1)
sobol_price_312_input = model_sobol_input_312500[sobol_price_312_filter, :]
sobol_price_312_output = sobol_price_312500[sobol_price_312_filter, :]

sobol2_price_100_filter = np.all(sobol2_price_100000 != 0, axis = 1)
sobol2_price_100_input = model_sobol2_input_100000[sobol2_price_100_filter, :]
sobol2_price_100_output = sobol2_price_100000[sobol2_price_100_filter, :]

sobol2_price_200_filter = np.all(sobol2_price_200000 != 0, axis = 1)
sobol2_price_200_input = model_sobol2_input_200000[sobol2_price_200_filter, :]
sobol2_price_200_output = sobol2_price_200000[sobol2_price_200_filter, :]

sobol2_price_312_filter = np.all(sobol2_price_312500 != 0, axis = 1)
sobol2_price_312_input = model_sobol2_input_312500[sobol2_price_312_filter, :]
sobol2_price_312_output = sobol2_price_312500[sobol2_price_312_filter, :]

# Single filtering
sobol2_single_price = sobol2_single_price_output[sobol2_single_price_output != 0]
sobol2_single_price_input = sobol2_single_input[sobol2_single_price_output != 0]
sobol2_single_imp_vol = sobol2_single_imp_vol_output[sobol2_single_imp_vol_output != 0]
sobol2_single_imp_vol_input = sobol2_single_input[sobol2_single_imp_vol_output != 0]

# shaping for NN's
sobol2_single_price_input = np.reshape(sobol2_single_price_input, (-1, 1))
sobol2_single_imp_vol_input = np.reshape(sobol2_single_imp_vol_input, (-1, 1))
sobol2_single_price = np.reshape(sobol2_single_price, (-1, 1))
sobol2_single_imp_vol = np.reshape(sobol2_single_imp_vol, (-1, 1))

"""
tanh_set = [
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 3, 50, "HestonSobolGridImpVol1_3_50", True, "tanh"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 3, 100, "HestonSobolGridImpVol1_3_100", True, "tanh"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 3, 500, "HestonSobolGridImpVol1_3_500", True, "tanh"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 3, 1000, "HestonSobolGridImpVol1_3_1000", True, "tanh"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 4, 50, "HestonSobolGridImpVol1_4_50", True, "tanh"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 4, 100, "HestonSobolGridImpVol1_4_100", True, "tanh"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 4, 500, "HestonSobolGridImpVol1_4_500", True, "tanh"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 4, 1000, "HestonSobolGridImpVol1_4_1000", True, "tanh"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 5, 50, "HestonSobolGridImpVol1_5_50", True, "tanh"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 5, 100, "HestonSobolGridImpVol1_5_100", True, "tanh"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 5, 500, "HestonSobolGridImpVol1_5_500", True, "tanh"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 5, 1000, "HestonSobolGridImpVol1_5_1000", True, "tanh"],
    [sobol_price_100_input, sobol_price_100_output, 3, 50, "HestonSobolGridPrice1_3_50", True, "tanh"],
    [sobol_price_100_input, sobol_price_100_output, 3, 100, "HestonSobolGridPrice1_3_100", True, "tanh"],
    [sobol_price_100_input, sobol_price_100_output, 3, 500, "HestonSobolGridPrice1_3_500", True, "tanh"],
    [sobol_price_100_input, sobol_price_100_output, 3, 1000, "HestonSobolGridPrice1_3_1000", True, "tanh"],
    [sobol_price_100_input, sobol_price_100_output, 4, 50, "HestonSobolGridPrice1_4_50", True, "tanh"],
    [sobol_price_100_input, sobol_price_100_output, 4, 100, "HestonSobolGridPrice1_4_100", True, "tanh"],
    [sobol_price_100_input, sobol_price_100_output, 4, 500, "HestonSobolGridPrice1_4_500", True, "tanh"],
    [sobol_price_100_input, sobol_price_100_output, 4, 1000, "HestonSobolGridPrice1_4_1000", True, "tanh"],
    [sobol_price_100_input, sobol_price_100_output, 5, 50, "HestonSobolGridPrice1_5_50", True, "tanh"],
    [sobol_price_100_input, sobol_price_100_output, 5, 100, "HestonSobolGridPrice1_5_100", True, "tanh"],
    [sobol_price_100_input, sobol_price_100_output, 5, 500, "HestonSobolGridPrice1_5_500", True, "tanh"],
    [sobol_price_100_input, sobol_price_100_output, 5, 1000, "HestonSobolGridPrice1_5_1000", True, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 3, 50, "HestonSobolGridImpVol3_3_50", True, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 3, 100, "HestonSobolGridImpVol3_3_100", True, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 3, 500, "HestonSobolGridImpVol3_3_500", True, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 3, 1000, "HestonSobolGridImpVol3_3_1000", True, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 4, 50, "HestonSobolGridImpVol3_4_50", True, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 4, 100, "HestonSobolGridImpVol3_4_100", True, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 4, 500, "HestonSobolGridImpVol3_4_500", True, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 4, 1000, "HestonSobolGridImpVol3_4_1000", True, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 5, 50, "HestonSobolGridImpVol3_5_50", True, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 5, 100, "HestonSobolGridImpVol3_5_100", True, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 5, 500, "HestonSobolGridImpVol3_5_500", True, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 5, 1000, "HestonSobolGridImpVol3_5_1000", True, "tanh"],
    [sobol_price_312_input, sobol_price_312_output, 3, 50, "HestonSobolGridPrice3_3_50", True, "tanh"],
    [sobol_price_312_input, sobol_price_312_output, 3, 100, "HestonSobolGridPrice3_3_100", True, "tanh"],
    [sobol_price_312_input, sobol_price_312_output, 3, 500, "HestonSobolGridPrice3_3_500", True, "tanh"],
    [sobol_price_312_input, sobol_price_312_output, 3, 1000, "HestonSobolGridPrice3_3_1000", True, "tanh"],
    [sobol_price_312_input, sobol_price_312_output, 4, 50, "HestonSobolGridPrice3_4_50", True, "tanh"],
    [sobol_price_312_input, sobol_price_312_output, 4, 100, "HestonSobolGridPrice3_4_100", True, "tanh"],
    [sobol_price_312_input, sobol_price_312_output, 4, 500, "HestonSobolGridPrice3_4_500", True, "tanh"],
    [sobol_price_312_input, sobol_price_312_output, 4, 1000, "HestonSobolGridPrice3_4_1000", True, "tanh"],
    [sobol_price_312_input, sobol_price_312_output, 5, 50, "HestonSobolGridPrice3_5_50", True, "tanh"],
    [sobol_price_312_input, sobol_price_312_output, 5, 100, "HestonSobolGridPrice3_5_100", True, "tanh"],
    [sobol_price_312_input, sobol_price_312_output, 5, 500, "HestonSobolGridPrice3_5_500", True, "tanh"],
    [sobol_price_312_input, sobol_price_312_output, 5, 1000, "HestonSobolGridPrice3_5_1000", True, "tanh"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 3, 500, "HestonGridImpVolFilter_3_500", True, "tanh"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 3, 1000, "HestonGridImpVolFilter_3_1000", True, "tanh"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 4, 500, "HestonGridImpVolFilter_4_500", True, "tanh"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 4, 1000, "HestonGridImpVolFilter_4_1000", True, "tanh"]
]

normal_out_set = [
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 3, 500, "HestonGridImpVolFilter_3_500", False],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 3, 1000, "HestonGridImpVolFilter_3_1000", False],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 4, 500, "HestonGridImpVolFilter_4_500", False],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 4, 1000, "HestonGridImpVolFilter_4_1000", False],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 3, 500, "HestonGridImpVolFilter_3_500", False, "tanh"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 3, 1000, "HestonGridImpVolFilter_3_1000", False, "tanh"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 4, 500, "HestonGridImpVolFilter_4_500", False, "tanh"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 4, 1000, "HestonGridImpVolFilter_4_1000", False, "tanh"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 3, 500, "HestonGridImpVolFilter_3_500", False, "mix"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 3, 1000, "HestonGridImpVolFilter_3_1000", False, "mix"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 4, 500, "HestonGridImpVolFilter_4_500", False, "mix"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 4, 1000, "HestonGridImpVolFilter_4_1000", False, "mix"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 3, 50, "HestonSobolGridImpVol3_3_50", False],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 3, 500, "HestonSobolGridImpVol3_3_500", False],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 4, 50, "HestonSobolGridImpVol3_4_50", False],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 4, 500, "HestonSobolGridImpVol3_4_500", False],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 5, 50, "HestonSobolGridImpVol3_5_50", False],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 5, 500, "HestonSobolGridImpVol3_5_500", False],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 3, 50, "HestonSobolGridImpVol3_3_50", False, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 3, 500, "HestonSobolGridImpVol3_3_500", False, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 4, 50, "HestonSobolGridImpVol3_4_50", False, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 4, 500, "HestonSobolGridImpVol3_4_500", False, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 5, 50, "HestonSobolGridImpVol3_5_50", False, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 5, 500, "HestonSobolGridImpVol3_5_500", False, "tanh"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 3, 50, "HestonSobolGridImpVol3_3_50", False, "mix"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 3, 500, "HestonSobolGridImpVol3_3_500", False, "mix"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 4, 50, "HestonSobolGridImpVol3_4_50", False, "mix"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 4, 500, "HestonSobolGridImpVol3_4_500", False, "mix"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 5, 50, "HestonSobolGridImpVol3_5_50", False, "mix"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 5, 500, "HestonSobolGridImpVol3_5_500", False, "mix"]
]

normalize_set = [
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 3, 100, "HestonSobolGridImpVol3_3_100", False, "normal", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 3, 1000, "HestonSobolGridImpVol3_3_1000", False, "normal", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 4, 100, "HestonSobolGridImpVol3_4_100", False, "normal", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 4, 1000, "HestonSobolGridImpVol3_4_1000", False, "normal", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 5, 100, "HestonSobolGridImpVol3_5_100", False, "normal", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 5, 1000, "HestonSobolGridImpVol3_5_1000", False, "normal", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 3, 100, "HestonSobolGridImpVol3_3_100", False, "tanh", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 3, 1000, "HestonSobolGridImpVol3_3_1000", False, "tanh", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 4, 100, "HestonSobolGridImpVol3_4_100", False, "tanh", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 4, 1000, "HestonSobolGridImpVol3_4_1000", False, "tanh", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 5, 100, "HestonSobolGridImpVol3_5_100", False, "tanh", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 5, 1000, "HestonSobolGridImpVol3_5_1000", False, "tanh", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 3, 100, "HestonSobolGridImpVol3_3_100", False, "mix", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 3, 1000, "HestonSobolGridImpVol3_3_1000", False, "mix", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 4, 100, "HestonSobolGridImpVol3_4_100", False, "mix", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 4, 1000, "HestonSobolGridImpVol3_4_1000", False, "mix", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 5, 100, "HestonSobolGridImpVol3_5_100", False, "mix", "normalize"],
    [sobol_imp_vol_312_input, sobol_imp_vol_312_output, 5, 1000, "HestonSobolGridImpVol3_5_1000", False, "mix", "normalize"]
]
"""

sobol2_set = [
    [sobol2_imp_vol_312_input, sobol2_imp_vol_312_output, 3, 100, "Sobol2_grid_imp_vol_312_3_100", False, "normal", "normalize"],
    [sobol2_imp_vol_312_input, sobol2_imp_vol_312_output, 3, 1000, "Sobol2_grid_imp_vol_312_3_1000", False, "normal", "normalize"],
    [sobol2_imp_vol_312_input, sobol2_imp_vol_312_output, 4, 100, "Sobol2_grid_imp_vol_312_4_100", False, "normal", "normalize"],
    [sobol2_imp_vol_312_input, sobol2_imp_vol_312_output, 4, 1000, "Sobol2_grid_imp_vol_312_4_1000", False, "normal", "normalize"],
    [sobol2_imp_vol_312_input, sobol2_imp_vol_312_output, 5, 100, "Sobol2_grid_imp_vol_312_5_100", False, "normal", "normalize"],
    [sobol2_imp_vol_312_input, sobol2_imp_vol_312_output, 5, 1000, "Sobol2_grid_imp_vol_312_5_1000", False, "normal", "normalize"],
    [sobol2_imp_vol_100_input, sobol2_imp_vol_100_output, 4, 100, "Sobol2_grid_imp_vol_100_4_100", False, "normal", "normalize"],
    [sobol2_imp_vol_100_input, sobol2_imp_vol_100_output, 4, 1000, "Sobol2_grid_imp_vol_100_4_1000", False, "normal", "normalize"],
]

wide_set = [
    [sobol2_price_312_input, sobol2_price_312_output, 3, 100, "Sobol2_grid_price_312_3_100", True, "normal", "standardize"],
    [sobol2_price_312_input, sobol2_price_312_output, 4, 100, "Sobol2_grid_price_312_4_100", True, "normal", "standardize"],
    [sobol2_price_312_input, sobol2_price_312_output, 5, 100, "Sobol2_grid_price_312_5_100", True, "normal", "standardize"],
    [sobol2_price_312_input, sobol2_price_312_output, 3, 500, "Sobol2_grid_price_312_3_500", True, "normal", "standardize"],
    [sobol2_price_312_input, sobol2_price_312_output, 4, 500, "Sobol2_grid_price_312_4_500", True, "normal", "standardize"],
    [sobol2_price_312_input, sobol2_price_312_output, 5, 500, "Sobol2_grid_price_312_5_500", True, "normal", "standardize"]
]

single_price_set = [
    [sobol2_single_price_input, sobol2_single_price, 3, 100, "sobol_single_price_3_100", False, "normal", "normalize"],
    [sobol2_single_price_input, sobol2_single_price, 3, 500, "sobol_single_price_3_500", False, "normal", "normalize"],
    [sobol2_single_price_input, sobol2_single_price, 4, 100, "sobol_single_price_4_100", False, "normal", "normalize"],
    [sobol2_single_price_input, sobol2_single_price, 4, 500, "sobol_single_price_4_500", False, "normal", "normalize"],
    [sobol2_single_price_input, sobol2_single_price, 5, 100, "sobol_single_price_5_100", False, "normal", "normalize"],
    [sobol2_single_price_input, sobol2_single_price, 5, 500, "sobol_single_price_5_500", False, "normal", "normalize"],
    [sobol2_single_imp_vol_input, sobol2_single_imp_vol, 3, 100, "sobol_single_imp_vol_3_100", False, "normal", "normalize"],
    [sobol2_single_imp_vol_input, sobol2_single_imp_vol, 3, 500, "sobol_single_imp_vol_3_500", False, "normal", "normalize"],
    [sobol2_single_imp_vol_input, sobol2_single_imp_vol, 4, 100, "sobol_single_imp_vol_4_100", False, "normal", "normalize"],
    [sobol2_single_imp_vol_input, sobol2_single_imp_vol, 4, 500, "sobol_single_imp_vol_4_500", False, "normal", "normalize"],
    [sobol2_single_imp_vol_input, sobol2_single_imp_vol, 5, 100, "sobol_single_imp_vol_5_100", False, "normal", "normalize"],
    [sobol2_single_imp_vol_input, sobol2_single_imp_vol, 5, 500, "sobol_single_imp_vol_5_500", False, "normal", "normalize"]
]

cpu_cores = 4
# parallel
pool = Pool(cpu_cores)
#res_sobol_normalize = pool.starmap(mg.NNModel, normalize_set)
#res_sobol_tanh = pool.starmap(mg.NNModel, tanh_set)
res_sobol2 = pool.starmap(mg.NNModel, wide_set)
print(res_sobol2)
print("Starting single sets")
res_single = pool.starmap(mg.NNModel, single_price_set)

#res_price_grid = pool.starmap(mg.NNModel, paral_price_grid)
#res_single = pool.starmap(mg.NNModel, paral_single)
#print(res_sobol_mix)
#print(res_sobol_tanh)
print(res_single)