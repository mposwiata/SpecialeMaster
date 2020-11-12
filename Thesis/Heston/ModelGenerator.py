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
wide_imp = np.loadtxt("Data/hestonGridImpVol_wide.csv", delimiter = ",")
wide_price = np.loadtxt("Data/hestonGridPrice_wide.csv", delimiter = ",")

### Sobol, wider
sobol_input = np.loadtxt("Data/hestonSobolGridInput2_279936.csv", delimiter = ",")
sobol_imp = np.loadtxt("Data/hestonSobolGridImpVol2_279936.csv", delimiter = ",")

### Sobol, 200.000
model_sobol2_input_200000 = np.loadtxt("Data/hestonSobolGridInput2_200000.csv", delimiter = ",")
sobol2_imp_vol_200000 = np.loadtxt("Data/hestonSobolGridImpVol2_200000.csv", delimiter = ",")

# Grid filtering, rows with 0 in
wide_imp_filter = np.all(wide_imp != 0, axis = 1)
model_wide_filter = model_wide[wide_imp_filter, :]
wide_imp_filter = wide_imp[wide_imp_filter, :]

wide_price_filter = np.all(wide_price != 0, axis = 1)
model_wide_price_filter = model_wide[wide_price_filter, :]
wide_price_filter = wide_price[wide_price_filter, :]

sobol_imp_filter = np.all(sobol_imp != 0, axis = 1)
sobol_input = sobol_input[sobol_imp_filter, :]
sobol_imp = sobol_imp[sobol_imp_filter, :]

sobol2_imp_vol_200_filter = np.all(sobol2_imp_vol_200000 != 0, axis = 1)
sobol2_imp_vol_200_input = model_sobol2_input_200000[sobol2_imp_vol_200_filter, :]
sobol2_imp_vol_200_output = sobol2_imp_vol_200000[sobol2_imp_vol_200_filter, :]

sobol_set = [
    [model_wide, wide_imp, 1, 500, "standard_all_1_500", False, "normal", "normalize"],
    [model_wide, wide_imp, 1, 1000, "standard_all_1_1000", False, "normal", "normalize"],
    [model_wide, wide_imp, 2, 500, "standard_all_2_500", False, "normal", "normalize"],
    [model_wide, wide_imp, 2, 1000, "standard_all_2_1000", False, "normal", "normalize"],
    [model_wide, wide_imp, 3, 500, "standard_all_3_500", False, "normal", "normalize"],
    [model_wide, wide_imp, 3, 1000, "standard_all_3_1000", False, "normal", "normalize"],
    [model_wide, wide_imp, 4, 500, "standard_all_4_500", False, "normal", "normalize"],
    [model_wide, wide_imp, 4, 1000, "standard_all_4_1000", False, "normal", "normalize"],
    [model_wide, wide_imp, 5, 500, "standard_all_5_500", False, "normal", "normalize"],
    [model_wide, wide_imp, 5, 1000, "standard_all_5_1000", False, "normal", "normalize"],
    [model_wide_price_filter, wide_price_filter, 1, 500, "standard_price_1_500", False, "normal", "normalize"],
    [model_wide_price_filter, wide_price_filter, 1, 1000, "standard_price_1_1000", False, "normal", "normalize"],
    [model_wide_price_filter, wide_price_filter, 2, 500, "standard_price_2_500", False, "normal", "normalize"],
    [model_wide_price_filter, wide_price_filter, 2, 1000, "standard_price_2_1000", False, "normal", "normalize"],
    [model_wide_price_filter, wide_price_filter, 3, 500, "standard_price_3_500", False, "normal", "normalize"],
    [model_wide_price_filter, wide_price_filter, 3, 1000, "standard_price_3_1000", False, "normal", "normalize"],
    [model_wide_price_filter, wide_price_filter, 4, 500, "standard_price_4_500", False, "normal", "normalize"],
    [model_wide_price_filter, wide_price_filter, 4, 1000, "standard_price_4_1000", False, "normal", "normalize"],
    [model_wide_price_filter, wide_price_filter, 5, 500, "standard_price_5_500", False, "normal", "normalize"],
    [model_wide_price_filter, wide_price_filter, 5, 1000, "standard_price_5_1000", False, "normal", "normalize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 500, "sobol_standard_200_3_500", False, "normal", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 1000, "sobol_standard_200_3_1000", False, "normal", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 500, "sobol_standard_200_4_500", False, "normal", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 1000, "sobol_standard_200_4_500", False, "normal", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 50, "sobol_standard_200_5_50", False, "normal", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 100, "sobol_standard_200_5_100", False, "normal", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 500, "sobol_standard_200_5_500", False, "normal", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 1000, "sobol_standard_200_5_500", False, "normal", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 500, "sobol_standard_200_3_500", False, "tanh", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 1000, "sobol_standard_200_3_1000", False, "tanh", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 500, "sobol_standard_200_4_500", False, "tanh", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 1000, "sobol_standard_200_4_500", False, "tanh", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 50, "sobol_standard_200_5_50", False, "tanh", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 100, "sobol_standard_200_5_100", False, "tanh", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 500, "sobol_standard_200_5_500", False, "tanh", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 1000, "sobol_standard_200_5_500", False, "tanh", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 500, "sobol_standard_200_3_500", False, "mix", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 3, 1000, "sobol_standard_200_3_1000", False, "mix", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 500, "sobol_standard_200_4_500", False, "mix", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 4, 1000, "sobol_standard_200_4_500", False, "mix", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 50, "sobol_standard_200_5_50", False, "mix", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 100, "sobol_standard_200_5_100", False, "mix", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 500, "sobol_standard_200_5_500", False, "mix", "standardize"],
    [sobol2_imp_vol_200_input, sobol2_imp_vol_200_output, 5, 1000, "sobol_standard_200_5_500", False, "mix", "standardize"]
]

cpu_cores = min(cpu_count(), len(sobol_set))
# parallel
pool = Pool(cpu_cores)
res_sobol = pool.starmap(mg.NNModel, sobol_set)
print(res_sobol)
