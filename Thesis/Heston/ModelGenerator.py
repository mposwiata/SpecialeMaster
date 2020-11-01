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

grid_price_output = np.loadtxt("Data/hestonGridPrice.csv", delimiter = ",")
grid_imp_vol_output = np.loadtxt("Data/hestonGridImpVol.csv", delimiter = ",")
sobol_price_100000 = np.loadtxt("Data/hestonSobolGridPrice_100000.csv", delimiter = ",")
sobol_price_200000 = np.loadtxt("Data/hestonSobolGridPrice_200000.csv", delimiter = ",")
sobol_imp_vol_100000 = np.loadtxt("Data/hestonSobolGridImpVol_100000.csv", delimiter = ",")
sobol_imp_vol_200000 = np.loadtxt("Data/hestonSobolGridImpVol_200000.csv", delimiter = ",")

"""
total_comb = np.shape(model_input)[0] * np.shape(option_input)[0]
total_cols = np.shape(model_input)[1] + np.shape(option_input)[1]
total_options = np.shape(option_input)[0]
singleInput = np.empty((total_comb, total_cols))
singlePrice_output = np.empty((total_comb, 1))
singleImpVol_output = np.empty((total_comb, 1))
for i in range(np.shape(model_input)[0]):
    for j in range(total_options):
        singleInput[i*total_options+j, 0:np.shape(model_input)[1]] = model_input[i]
        singleInput[i*total_options+j, (np.shape(model_input)[1]) : total_cols] = option_input[j]
        singlePrice_output[i*total_options+j] = gridPriceOutput[i, j]
        singleImpVol_output[i*total_options+j] = gridImpVolOutput[i, j]

singlePrice_output = singlePrice_output.flatten()
singleImpVol_output = singleImpVol_output.flatten()
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

sobol_price_100_filter = np.all(sobol_price_100000 != 0, axis = 1)
sobol_price_100_input = model_sobol_input_100000[sobol_price_100_filter, :]
sobol_price_100_output = sobol_price_100000[sobol_price_100_filter, :]

sobol_price_200_filter = np.all(sobol_price_200000 != 0, axis = 1)
sobol_price_200_input = model_sobol_input_200000[sobol_price_200_filter, :]
sobol_price_200_output = sobol_price_200000[sobol_price_200_filter, :]

paral = [
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 3, 50, "HestonSobolGridImpVol1_3_50"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 3, 100, "HestonSobolGridImpVol1_3_100"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 3, 500, "HestonSobolGridImpVol1_3_500"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 3, 1000, "HestonSobolGridImpVol1_3_1000"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 4, 50, "HestonSobolGridImpVol1_4_50"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 4, 100, "HestonSobolGridImpVol1_4_100"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 4, 500, "HestonSobolGridImpVol1_4_500"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 4, 1000, "HestonSobolGridImpVol1_4_1000"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 5, 50, "HestonSobolGridImpVol1_5_50"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 5, 100, "HestonSobolGridImpVol1_5_100"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 5, 500, "HestonSobolGridImpVol1_5_500"],
    [sobol_imp_vol_100_input, sobol_imp_vol_100_output, 5, 1000, "HestonSobolGridImpVol1_5_1000"],
    [sobol_price_100_input, sobol_price_100_output, 3, 50, "HestonSobolGridPrice1_3_50"],
    [sobol_price_100_input, sobol_price_100_output, 3, 100, "HestonSobolGridPrice1_3_100"],
    [sobol_price_100_input, sobol_price_100_output, 3, 500, "HestonSobolGridPrice1_3_500"],
    [sobol_price_100_input, sobol_price_100_output, 3, 1000, "HestonSobolGridPrice1_3_1000"],
    [sobol_price_100_input, sobol_price_100_output, 4, 50, "HestonSobolGridPrice1_4_50"],
    [sobol_price_100_input, sobol_price_100_output, 4, 100, "HestonSobolGridPrice1_4_100"],
    [sobol_price_100_input, sobol_price_100_output, 4, 500, "HestonSobolGridPrice1_4_500"],
    [sobol_price_100_input, sobol_price_100_output, 4, 1000, "HestonSobolGridPrice1_4_1000"],
    [sobol_price_100_input, sobol_price_100_output, 5, 50, "HestonSobolGridPrice1_5_50"],
    [sobol_price_100_input, sobol_price_100_output, 5, 100, "HestonSobolGridPrice1_5_100"],
    [sobol_price_100_input, sobol_price_100_output, 5, 500, "HestonSobolGridPrice1_5_500"],
    [sobol_price_100_input, sobol_price_100_output, 5, 1000, "HestonSobolGridPrice1_5_1000"],
    [sobol_imp_vol_200_input, sobol_imp_vol_200_output, 3, 50, "HestonSobolGridImpVol2_3_50"],
    [sobol_imp_vol_200_input, sobol_imp_vol_200_output, 3, 100, "HestonSobolGridImpVol2_3_100"],
    [sobol_imp_vol_200_input, sobol_imp_vol_200_output, 3, 500, "HestonSobolGridImpVol2_3_500"],
    [sobol_imp_vol_200_input, sobol_imp_vol_200_output, 3, 1000, "HestonSobolGridImpVol2_3_1000"],
    [sobol_imp_vol_200_input, sobol_imp_vol_200_output, 4, 50, "HestonSobolGridImpVol2_4_50"],
    [sobol_imp_vol_200_input, sobol_imp_vol_200_output, 4, 100, "HestonSobolGridImpVol2_4_100"],
    [sobol_imp_vol_200_input, sobol_imp_vol_200_output, 4, 500, "HestonSobolGridImpVol2_4_500"],
    [sobol_imp_vol_200_input, sobol_imp_vol_200_output, 4, 1000, "HestonSobolGridImpVol2_4_1000"],
    [sobol_imp_vol_200_input, sobol_imp_vol_200_output, 5, 50, "HestonSobolGridImpVol2_5_50"],
    [sobol_imp_vol_200_input, sobol_imp_vol_200_output, 5, 100, "HestonSobolGridImpVol2_5_100"],
    [sobol_imp_vol_200_input, sobol_imp_vol_200_output, 5, 500, "HestonSobolGridImpVol2_5_500"],
    [sobol_imp_vol_200_input, sobol_imp_vol_200_output, 5, 1000, "HestonSobolGridImpVol2_5_1000"],
    [sobol_price_200_input, sobol_price_200_output, 3, 50, "HestonSobolGridPrice2_3_50"],
    [sobol_price_200_input, sobol_price_200_output, 3, 100, "HestonSobolGridPrice2_3_100"],
    [sobol_price_200_input, sobol_price_200_output, 3, 500, "HestonSobolGridPrice2_3_500"],
    [sobol_price_200_input, sobol_price_200_output, 3, 1000, "HestonSobolGridPrice2_3_1000"],
    [sobol_price_200_input, sobol_price_200_output, 4, 50, "HestonSobolGridPrice2_4_50"],
    [sobol_price_200_input, sobol_price_200_output, 4, 100, "HestonSobolGridPrice2_4_100"],
    [sobol_price_200_input, sobol_price_200_output, 4, 500, "HestonSobolGridPrice2_4_500"],
    [sobol_price_200_input, sobol_price_200_output, 4, 1000, "HestonSobolGridPrice2_4_1000"],
    [sobol_price_200_input, sobol_price_200_output, 5, 50, "HestonSobolGridPrice2_5_50"],
    [sobol_price_200_input, sobol_price_200_output, 5, 100, "HestonSobolGridPrice2_5_100"],
    [sobol_price_200_input, sobol_price_200_output, 5, 500, "HestonSobolGridPrice2_5_500"],
    [sobol_price_200_input, sobol_price_200_output, 5, 1000, "HestonSobolGridPrice2_5_1000"]
]
cpu_cores = cpu_count()
# parallel
pool = Pool(cpu_cores)
res = pool.starmap(mg.NNModelTanh, paral)
print(res)


"""
# Single filtering
filteredPriceSingle = singlePrice_output[singlePrice_output != 0]
filteredSingleModelPrice = singleInput[singlePrice_output != 0]
filteredImpVolSingle = singleImpVol_output[singleImpVol_output != 0]
filteredSingleModelImpVol = singleInput[singleImpVol_output != 0]

# shaping for NN's
singlePrice_output = np.reshape(singlePrice_output, (-1, 1))
singleImpVol_output = np.reshape(singleImpVol_output, (-1, 1))
filteredPriceSingle = np.reshape(filteredPriceSingle, (-1, 1))
filteredImpVolSingle = np.reshape(filteredImpVolSingle, (-1, 1))
"""

"""
cpu_cores = cpu_count() / 2
parallel_list = [
    [model_input, grid_imp_vol_output, 1, 50, "HestonGridImpVolAll_1_50"],
    [model_input, grid_imp_vol_output, 1, 100, "HestonGridImpVolAll_1_100"],
    [model_input, grid_imp_vol_output, 1, 500, "HestonGridImpVolAll_1_500"],
    [model_input, grid_imp_vol_output, 1, 1000, "HestonGridImpVolAll_1_1000"],
    [model_input, grid_imp_vol_output, 2, 50, "HestonGridImpVolAll_2_50"],
    [model_input, grid_imp_vol_output, 2, 100, "HestonGridImpVolAll_2_100"],
    [model_input, grid_imp_vol_output, 2, 500, "HestonGridImpVolAll_2_500"],
    [model_input, grid_imp_vol_output, 2, 1000, "HestonGridImpVolAll_2_1000"],
    [model_input, grid_imp_vol_output, 3, 50, "HestonGridImpVolAll_3_50"],
    [model_input, grid_imp_vol_output, 3, 100, "HestonGridImpVolAll_3_100"],
    [model_input, grid_imp_vol_output, 3, 500, "HestonGridImpVolAll_3_500"],
    [model_input, grid_imp_vol_output, 3, 1000, "HestonGridImpVolAll_3_1000"],
    [model_input, grid_imp_vol_output, 4, 50, "HestonGridImpVolAll_4_50"],
    [model_input, grid_imp_vol_output, 4, 100, "HestonGridImpVolAll_4_100"],
    [model_input, grid_imp_vol_output, 4, 500, "HestonGridImpVolAll_4_500"],
    [model_input, grid_imp_vol_output, 4, 1000, "HestonGridImpVolAll_4_1000"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 1, 50, "HestonGridImpVolFilter_1_50"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 1, 100, "HestonGridImpVolFilter_1_100"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 1, 500, "HestonGridImpVolFilter_1_500"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 1, 1000, "HestonGridImpVolFilter_1_1000"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 2, 50, "HestonGridImpVolFilter_2_50"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 2, 100, "HestonGridImpVolFilter_2_100"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 2, 500, "HestonGridImpVolFilter_2_500"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 2, 1000, "HestonGridImpVolFilter_2_1000"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 3, 50, "HestonGridImpVolFilter_3_50"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 3, 100, "HestonGridImpVolFilter_3_100"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 3, 500, "HestonGridImpVolFilter_3_500"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 3, 1000, "HestonGridImpVolFilter_3_1000"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 4, 50, "HestonGridImpVolFilter_4_50"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 4, 100, "HestonGridImpVolFilter_4_100"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 4, 500, "HestonGridImpVolFilter_4_500"],
    [filtered_grid_model_imp_vol, filtered_grid_imp_vol, 4, 1000, "HestonGridImpVolFilter_4_1000"]
]

# parallel
pool = Pool(cpu_cores)
res = pool.starmap(mg.NNModel, parallel_list)
print(res)
"""