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

model_input = dg.modelInputGenerator() #faster than reading file
option_input = dg.optionInputGenerator()

gridPriceOutput = np.loadtxt("Data/hestonGridPrice.csv", delimiter = ",")
gridImpVolOutput = np.loadtxt("Data/hestonGridImpVol.csv", delimiter = ",")

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

# Grid filtering, rows with 0 in
filterPriceGrid = np.all(gridPriceOutput != 0, axis = 1)
filteredGridModelPrice = model_input[filterPriceGrid, :]
filteredGridPrice = gridPriceOutput[filterPriceGrid, :]

filterImpVolGrid = np.all(gridImpVolOutput != 0, axis = 1)
filteredGridModelImpVol = model_input[filterImpVolGrid, :]
filteredGridImpVol = gridImpVolOutput[filterImpVolGrid, :]

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

cpu_cores = cpu_count()
parallel_list = [
    [model_input, gridPriceOutput, 4, 100, "HestonGridPriceAll_100"],
    [model_input, gridPriceOutput, 4, 1000, "HestonGridPriceAll_1000"],
    [model_input, gridImpVolOutput, 4, 100, "HestonGridImpVolAll_100"],
    [model_input, gridImpVolOutput, 4, 1000, "HestonGridImpVolAll_1000"],
    [filteredGridModelPrice, filteredGridPrice, 4, 100, "HestonGridPriceFilter_100"],
    [filteredGridModelPrice, filteredGridPrice, 4, 1000, "HestonGridPriceFilter_1000"],
    [filteredGridModelImpVol, filteredGridImpVol, 4, 100, "HestonGridImpVolFilter_100"],
    [filteredGridModelImpVol, filteredGridImpVol, 4, 1000, "HestonGridImpVolFilter_1000"],
    [singleInput, singlePrice_output, 4, 100, "HestonSinglePriceAll_100"],
    [singleInput, singlePrice_output, 4, 1000, "HestonSinglePriceAll_1000"],
    [singleInput, singleImpVol_output, 4, 100, "HestonSingleImpVolAll_100"],
    [singleInput, singleImpVol_output, 4, 1000, "HestonSingleImpVolAll_1000"],
    [filteredSingleModelPrice, filteredPriceSingle, 4, 100, "HestonSinglePriceFilter_100"],
    [filteredSingleModelPrice, filteredPriceSingle, 4, 1000, "HestonSinglePriceFilter_1000"],
    [filteredSingleModelImpVol, filteredImpVolSingle, 4, 100, "HestonSingleImpVolFilter_100"],
    [filteredSingleModelImpVol, filteredImpVolSingle, 4, 1000, "HestonSingleImpVolFilter_1000"]
]

# parallel
pool = Pool(cpu_cores)
res = pool.starmap(mg.NNModel, parallel_list)
print(res)

"""
processes = [
    Process(target = mg.NNModel, args=(model_input, gridPriceOutput, 4, 100, "HestonGridPriceAll_100")),
    Process(target = mg.NNModel, args=(model_input, gridPriceOutput, 4, 1000, "HestonGridPriceAll_1000")),
    Process(target = mg.NNModel, args=(model_input, gridImpVolOutput, 4, 100, "HestonGridImpVolAll_100")),
    Process(target = mg.NNModel, args=(model_input, gridImpVolOutput, 4, 1000, "HestonGridImpVolAll_100")),
    Process(target = mg.NNModel, args=(filteredGridModelPrice, filteredGridPrice, 4, 100, "HestonGridPriceFilter_100")),
    Process(target = mg.NNModel, args=(filteredGridModelPrice, filteredGridPrice, 4, 1000, "HestonGridPriceFilter_1000")),
    Process(target = mg.NNModel, args=(filteredGridModelImpVol, filteredGridImpVol, 4, 100, "HestonGridImpVolFilter_100")),
    Process(target = mg.NNModel, args=(filteredGridModelImpVol, filteredGridImpVol, 4, 1000, "HestonGridImpVolFilter_1000")),
    Process(target = mg.NNModel, args=(singleInput, gridPriceOutput, 4, 100, "HestonSinglePriceAll_100")),
    Process(target = mg.NNModel, args=(singleInput, gridPriceOutput, 4, 1000, "HestonSinglePriceAll_1000")),
    Process(target = mg.NNModel, args=(singleInput, gridImpVolOutput, 4, 100, "HestonSingleImpVolAll_100")),
    Process(target = mg.NNModel, args=(singleInput, gridImpVolOutput, 4, 1000, "HestonSingleImpVolAll_100")),
    Process(target = mg.NNModel, args=(filteredSingleModelPrice, filteredGridPrice, 4, 100, "HestonSinglePriceFilter_100")),
    Process(target = mg.NNModel, args=(filteredSingleModelPrice, filteredGridPrice, 4, 1000, "HestonSinglePriceFilter_1000")),
    Process(target = mg.NNModel, args=(filteredSingleModelImpVol, filteredGridImpVol, 4, 100, "HestonSingleImpVolFilter_100")),
    Process(target = mg.NNModel, args=(filteredSingleModelImpVol, filteredGridImpVol, 4, 1000, "HestonSingleImpVolFilter_1000"))
]

for p in processes:
    p.start()

results = [output.get() for p in p in processes]

for process in processes:
        process.join()

"""