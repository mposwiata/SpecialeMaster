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

### Wide model
model_wide = dg.model_input_generator()

### Sobol, wider
model_sobol2_input_200000 = np.loadtxt("Data/hestonSobolGridInput2_200000.csv", delimiter = ",")
sobol2_imp_vol_200000 = np.loadtxt("Data/hestonSobolGridImpVol2_200000.csv", delimiter = ",")

### Single prices on sobol2 312500
model_input = model_sobol2_input_200000
option_input = dg.option_input_generator()

total_comb = np.shape(model_input)[0] * np.shape(option_input)[0]
total_cols = np.shape(model_input)[1] + np.shape(option_input)[1]
total_options = np.shape(option_input)[0]
sobol2_single_input = np.empty((total_comb, total_cols))
sobol2_single_imp_vol_output = np.empty((total_comb, 1))
for i in range(np.shape(model_input)[0]):
    for j in range(total_options):
        sobol2_single_input[i*total_options+j, 0:np.shape(model_input)[1]] = model_input[i]
        sobol2_single_input[i*total_options+j, (np.shape(model_input)[1]) : total_cols] = option_input[j]
        sobol2_single_imp_vol_output[i*total_options+j] = sobol2_imp_vol_200000[i, j]

sobol2_single_imp_vol_output = sobol2_single_imp_vol_output.flatten()
model_input = dg.model_input_generator_old() #faster than reading file


# Grid filtering, rows with 0 in
sobol2_imp_vol_200_filter = np.all(sobol2_imp_vol_200000 != 0, axis = 1)
sobol2_imp_vol_200_input = model_sobol2_input_200000[sobol2_imp_vol_200_filter, :]
sobol2_imp_vol_200_output = sobol2_imp_vol_200000[sobol2_imp_vol_200_filter, :]

# Single filtering
sobol2_single_imp_vol = sobol2_single_imp_vol_output[sobol2_single_imp_vol_output != 0]
sobol2_single_imp_vol_input = sobol2_single_input[sobol2_single_imp_vol_output != 0, :]

# shaping for NN's
sobol2_single_imp_vol = np.reshape(sobol2_single_imp_vol, (-1, 1))

paral_set = [
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 1, 500, "sobol_200_1_500", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 1, 1000, "sobol_200_1_1000", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 2, 500, "sobol_200_2_500", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 2, 1000, "sobol_200_2_1000", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 50, "sobol_200_3_50", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 100, "sobol_200_3_100", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 500, "sobol_200_3_500", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 1000, "sobol_200_3_1000", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 50, "sobol_200_4_50", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 100, "sobol_200_4_100", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 500, "sobol_200_4_500", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 1000, "sobol_200_4_1000", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 50, "sobol_200_5_50", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 100, "sobol_200_5_100", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 500, "sobol_200_5_500", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 1000, "sobol_200_5_1000", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 500, "sobol_200_3_500", False, "tanh", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 1000, "sobol_200_3_1000", False, "tanh", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 500, "sobol_200_4_500", False, "tanh", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 1000, "sobol_200_4_500", False, "tanh", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 50, "sobol_200_5_50", False, "tanh", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 100, "sobol_200_5_100", False, "tanh", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 500, "sobol_200_5_500", False, "tanh", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 1000, "sobol_200_5_500", False, "tanh", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 500, "sobol_200_3_500", False, "mix", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 1000, "sobol_200_3_1000", False, "mix", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 500, "sobol_200_4_500", False, "mix", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 1000, "sobol_200_4_500", False, "mix", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 50, "sobol_200_5_50", False, "mix", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 100, "sobol_200_5_100", False, "mix", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 500, "sobol_200_5_500", False, "mix", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 1000, "sobol_200_5_500", False, "mix", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 500, "sobol_200_3_500", True, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 1000, "sobol_200_3_1000", True, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 500, "sobol_200_4_500", True, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 1000, "sobol_200_4_500", True, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 50, "sobol_200_5_50", True, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 100, "sobol_200_5_100", True, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 500, "sobol_200_5_500", True, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 1000, "sobol_200_5_500", True, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 500, "sobol_200_3_500", True, "tanh", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 1000, "sobol_200_3_1000", True, "tanh", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 500, "sobol_200_4_500", True, "tanh", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 1000, "sobol_200_4_500", True, "tanh", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 50, "sobol_200_5_50", True, "tanh", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 100, "sobol_200_5_100", True, "tanh", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 500, "sobol_200_5_500", True, "tanh", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 1000, "sobol_200_5_500", True, "tanh", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 500, "sobol_200_3_500", True, "mix", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 1000, "sobol_200_3_1000", True, "mix", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 500, "sobol_200_4_500", True, "mix", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 1000, "sobol_200_4_500", True, "mix", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 50, "sobol_200_5_50", True, "mix", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 100, "sobol_200_5_100", True, "mix", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 500, "sobol_200_5_500", True, "mix", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 1000, "sobol_200_5_500", True, "mix", "normalize"]
]

single_price_set = [
    [sobol2_single_imp_vol_input, sobol2_single_imp_vol, 3, 100, "sobol_single_imp_vol_3_100", False, "normal", "normalize"],
    [sobol2_single_imp_vol_input, sobol2_single_imp_vol, 3, 500, "sobol_single_imp_vol_3_500", False, "normal", "normalize"],
    [sobol2_single_imp_vol_input, sobol2_single_imp_vol, 4, 100, "sobol_single_imp_vol_4_100", False, "normal", "normalize"],
    [sobol2_single_imp_vol_input, sobol2_single_imp_vol, 4, 500, "sobol_single_imp_vol_4_500", False, "normal", "normalize"],
    [sobol2_single_imp_vol_input, sobol2_single_imp_vol, 5, 100, "sobol_single_imp_vol_5_100", False, "normal", "normalize"],
    [sobol2_single_imp_vol_input, sobol2_single_imp_vol, 5, 500, "sobol_single_imp_vol_5_500", False, "normal", "normalize"]
]

cpu_cores = min(cpu_count(), single_price_set)
# parallel
pool = Pool(cpu_cores)
#res_sobol = pool.starmap(mg.NNModel, paral_set)
res_single = pool.starmap(mg.NNModel, single_price_set)
#print(res_sobol)
print(res_single)
