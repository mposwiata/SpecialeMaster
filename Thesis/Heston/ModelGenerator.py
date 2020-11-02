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

paral_single = [
    [single_input, single_price_output, 3, 500, "HestonSinglePriceAll_3_500"],
    [single_input, single_price_output, 3, 1000, "HestonSinglePriceAll_3_1000"],
    [single_input, single_price_output, 4, 500, "HestonSinglePriceAll_4_500"],
    [single_input, single_price_output, 4, 1000, "HestonSinglePriceAll_4_1000"],
    [single_input, single_imp_vol_output, 3, 500, "HestonSingleImpVolAll_3_500"],
    [single_input, single_imp_vol_output, 3, 1000, "HestonSingleImpVolAll_3_1000"],
    [single_input, single_imp_vol_output, 4, 500, "HestonSingleImpVolAll_4_500"],
    [single_input, single_imp_vol_output, 4, 1000, "HestonSingleImpVolAll_4_1000"],
    [filtered_single_model_price, filtered_price_single, 3, 500, "HestonSinglePriceFilter_3_500"],
    [filtered_single_model_price, filtered_price_single, 3, 1000, "HestonSinglePriceFilter_3_1000"],
    [filtered_single_model_price, filtered_price_single, 4, 500, "HestonSinglePriceFilter_4_500"],
    [filtered_single_model_price, filtered_price_single, 4, 1000, "HestonSinglePriceFilter_4_1000"],
    [filtered_single_model_imp_vol, filtered_imp_vol_single, 3, 500, "HestonSingleImpVolFilter_3_500"],
    [filtered_single_model_imp_vol, filtered_imp_vol_single, 3, 1000, "HestonSingleImpVolFilter_3_1000"],
    [filtered_single_model_imp_vol, filtered_imp_vol_single, 4, 500, "HestonSingleImpVolFilter_4_500"],
    [filtered_single_model_imp_vol, filtered_imp_vol_single, 4, 1000, "HestonSingleImpVolFilter_4_1000"]
]

paral_price_grid = [
    [model_input, grid_price_output, 1, 50, "HestonGridPriceAll_1_50"],
    [model_input, grid_price_output, 1, 100, "HestonGridPriceAll_1_100"],
    [model_input, grid_price_output, 1, 500, "HestonGridPriceAll_1_500"],
    [model_input, grid_price_output, 1, 1000, "HestonGridPriceAll_1_1000"],
    [model_input, grid_price_output, 2, 50, "HestonGridPriceAll_2_50"],
    [model_input, grid_price_output, 2, 100, "HestonGridPriceAll_2_100"],
    [model_input, grid_price_output, 2, 500, "HestonGridPriceAll_2_500"],
    [model_input, grid_price_output, 2, 1000, "HestonGridPriceAll_2_1000"],
    [model_input, grid_price_output, 3, 50, "HestonGridPriceAll_3_50"],
    [model_input, grid_price_output, 3, 100, "HestonGridPriceAll_3_100"],
    [model_input, grid_price_output, 3, 500, "HestonGridPriceAll_3_500"],
    [model_input, grid_price_output, 3, 1000, "HestonGridPriceAll_3_1000"],
    [model_input, grid_price_output, 4, 50, "HestonGridPriceAll_4_50"],
    [model_input, grid_price_output, 4, 100, "HestonGridPriceAll_4_100"],
    [model_input, grid_price_output, 4, 500, "HestonGridPriceAll_4_500"],
    [model_input, grid_price_output, 4, 1000, "HestonGridPriceAll_4_1000"],
    [filtered_grid_model_price, filtered_grid_price, 1, 50, "HestonGridPriceFilter_1_50"],
    [filtered_grid_model_price, filtered_grid_price, 1, 100, "HestonGridPriceFilter_1_100"],
    [filtered_grid_model_price, filtered_grid_price, 1, 500, "HestonGridPriceFilter"],
    [filtered_grid_model_price, filtered_grid_price, 1, 1000, "HestonGridPriceFilter_1_1000"],
    [filtered_grid_model_price, filtered_grid_price, 2, 50, "HestonGridPriceFilter_2_50"],
    [filtered_grid_model_price, filtered_grid_price, 2, 100, "HestonGridPriceFilter_2_100"],
    [filtered_grid_model_price, filtered_grid_price, 2, 500, "HestonGridPriceFilter_2_500"],
    [filtered_grid_model_price, filtered_grid_price, 2, 1000, "HestonGridPriceFilter_2_1000"],
    [filtered_grid_model_price, filtered_grid_price, 3, 50, "HestonGridPriceFilter_3_50"],
    [filtered_grid_model_price, filtered_grid_price, 3, 100, "HestonGridPriceFilter_3_100"],
    [filtered_grid_model_price, filtered_grid_price, 3, 500, "HestonGridPriceFilter_3_500"],
    [filtered_grid_model_price, filtered_grid_price, 3, 1000, "HestonGridPriceFilter_3_1000"],
    [filtered_grid_model_price, filtered_grid_price, 4, 50, "HestonGridPriceFilter_4_50"],
    [filtered_grid_model_price, filtered_grid_price, 4, 100, "HestonGridPriceFilter_4_100"],
    [filtered_grid_model_price, filtered_grid_price, 4, 500, "HestonGridPriceFilter_4_500"],
    [filtered_grid_model_price, filtered_grid_price, 4, 1000, "HestonGridPriceFilter_4_1000"]
]

paral_sobol = [
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

"""

paral_sobol = [
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
    [sobol_price_100_input, sobol_price_100_output, 5, 1000, "HestonSobolGridPrice1_5_1000"]
]

cpu_cores = cpu_count()
# parallel
pool = Pool(cpu_cores)
res_sobol = pool.starmap(mg.NNModel, paral_sobol)
res_sobol_tanh = pool.starmap(mg.NNModelTanh, paral_sobol)
res_price_grid = pool.starmap(mg.NNModel, paral_price_grid)
#res_single = pool.starmap(mg.NNModel, paral_single)
print(res_sobol)
print(res_sobol_tanh)
print(res_price_grid)