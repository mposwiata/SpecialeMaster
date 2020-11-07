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

model_input = dg.model_input_generator() #faster than reading file
option_input = dg.option_input_generator()
model_sobol_input_100000 = np.loadtxt("Data/hestonSobolGridInput_100000.csv", delimiter = ",")
model_sobol_input_200000 = np.loadtxt("Data/hestonSobolGridInput_200000.csv", delimiter = ",")
model_sobol_input_312500 = np.loadtxt("Data/hestonSobolGridInput_312500.csv", delimiter = ",")

grid_price_output = np.loadtxt("Data/hestonGridPrice.csv", delimiter = ",")
grid_imp_vol_output = np.loadtxt("Data/hestonGridImpVol.csv", delimiter = ",")
sobol_price_100000 = np.loadtxt("Data/hestonSobolGridPrice_100000.csv", delimiter = ",")
sobol_price_200000 = np.loadtxt("Data/hestonSobolGridPrice_200000.csv", delimiter = ",")
sobol_price_312500 = np.loadtxt("Data/hestonSobolGridPrice_312500.csv", delimiter = ",")
sobol_imp_vol_100000 = np.loadtxt("Data/hestonSobolGridImpVol_100000.csv", delimiter = ",")
sobol_imp_vol_200000 = np.loadtxt("Data/hestonSobolGridImpVol_200000.csv", delimiter = ",")
sobol_imp_vol_312500 = np.loadtxt("Data/hestonSobolGridImpVol_312500.csv", delimiter = ",")

"""
total_comb = np.shape(model_input)[0] * np.shape(option_input)[0]
total_cols = np.shape(model_input)[1] + np.shape(option_input)[1]
total_options = np.shape(option_input)[0]
single_input = np.empty((total_comb, total_cols))
single_price_output = np.empty((total_comb, 1))
single_imp_vol_output = np.empty((total_comb, 1))
for i in range(np.shape(model_input)[0]):
    for j in range(total_options):
        single_input[i*total_options+j, 0:np.shape(model_input)[1]] = model_input[i]
        single_input[i*total_options+j, (np.shape(model_input)[1]) : total_cols] = option_input[j]
        single_price_output[i*total_options+j] = grid_price_output[i, j]
        single_imp_vol_output[i*total_options+j] = grid_imp_vol_output[i, j]

single_price_output = single_price_output.flatten()
single_imp_vol_output = single_imp_vol_output.flatten()
"""

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

sobol_price_100_filter = np.all(sobol_price_100000 != 0, axis = 1)
sobol_price_100_input = model_sobol_input_100000[sobol_price_100_filter, :]
sobol_price_100_output = sobol_price_100000[sobol_price_100_filter, :]

sobol_price_200_filter = np.all(sobol_price_200000 != 0, axis = 1)
sobol_price_200_input = model_sobol_input_200000[sobol_price_200_filter, :]
sobol_price_200_output = sobol_price_200000[sobol_price_200_filter, :]

sobol_price_312_filter = np.all(sobol_price_312500 != 0, axis = 1)
sobol_price_312_input = model_sobol_input_312500[sobol_price_312_filter, :]
sobol_price_312_output = sobol_price_312500[sobol_price_312_filter, :]

"""
# Single filtering
filtered_price_single = single_price_output[single_price_output != 0]
filtered_single_model_price = single_input[single_price_output != 0]
filtered_imp_vol_single = single_imp_vol_output[single_imp_vol_output != 0]
filtered_single_model_imp_vol = single_input[single_imp_vol_output != 0]

# shaping for NN's
single_price_output = np.reshape(single_price_output, (-1, 1))
single_imp_vol_output = np.reshape(single_imp_vol_output, (-1, 1))
filtered_price_single = np.reshape(filtered_price_single, (-1, 1))
filtered_imp_vol_single = np.reshape(filtered_imp_vol_single, (-1, 1))
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

cpu_cores = 4
# parallel
pool = Pool(cpu_cores)
#res_sobol_normalize = pool.starmap(mg.NNModel, normalize_set)
#res_sobol_tanh = pool.starmap(mg.NNModel, tanh_set)
res_sobol_norm = pool.starmap(mg.NNModel, normal_out_set)

#res_price_grid = pool.starmap(mg.NNModel, paral_price_grid)
#res_single = pool.starmap(mg.NNModel, paral_single)
#print(res_sobol_mix)
#print(res_sobol_tanh)
print(res_sobol_norm)