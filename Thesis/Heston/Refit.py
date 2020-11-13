import numpy as np
import tensorflow as tf 
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from multiprocess import Pool, cpu_count, Process
from sklearn.model_selection import train_test_split
from keras.models import load_model
from itertools import repeat
from functools import partial
import os
import joblib
import sys
import glob
sys.path.append(os.getcwd()) # added for calc server support

import time
from Thesis.Heston import DataGeneration as dg

def lr_schedule(epoch, rate):
    lower_lr = 1e-4
    upper_lr = lower_lr * 100
    no_epochs = 100
    peak_epoch = 45
    if epoch <= peak_epoch:
        lr = lower_lr + epoch / peak_epoch * (upper_lr - lower_lr)
    elif peak_epoch < epoch < peak_epoch * 2:
        lr = upper_lr - (epoch - peak_epoch) / peak_epoch * (upper_lr - lower_lr)
    else:
        lr = lower_lr * (1 - (epoch - 2 * peak_epoch) / (no_epochs - 2 * peak_epoch)) * (1 - 1 / 10)

    return lr

def refit_model(model_string : str, data : list):
    if os.path.exists(model_string.replace("Models2", "Models3")):
        return 0
    model = load_model(model_string)
    ending = model_string[-4:-3]

    X_train = data[0]
    X_test = data[1]
    Y_train = data[2]
    Y_test = data[3]

    norm_feature = joblib.load(model_string[:-5]+"_norm_features_"+ending+".pkl")
    X_train = norm_feature.fit_transform(X_train)
    X_test = norm_feature.transform(X_test)
    try:
        norm_labels = joblib.load(model_string[:-5]+"_norm_labels_"+ending+".pkl")
        Y_train = norm_labels.fit_transform(Y_train)
        Y_test = norm_labels.transform(Y_test)
    except:
        pass
    
    model_path = model_string[:model_string.rfind("/")]
    model_path = model_path.replace("Models2", "Models3")
    model_name = model_string[model_string.rfind("/") + 1:]

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_save = model_string.replace("Models2", "Models3")

    variance_scaler = tf.keras.initializers.VarianceScaling()

    ### Re initialize weights
    weights = [variance_scaler(shape = [*w.shape]) for w in model.get_weights()]

    model.set_weights(weights)

    adam = Adam()

    model.compile(
        loss = 'mean_squared_error', #mean squared error
        optimizer = adam
        )

    callbacks_list = [
        LearningRateScheduler(lr_schedule, verbose = 0),
        ModelCheckpoint(model_save, monitor="val_loss", save_best_only=True)
    ]

    start_time = time.time()
    model.fit(X_train, Y_train, epochs=100, batch_size=1024, verbose = 0, callbacks = callbacks_list, validation_split = 0.1, shuffle=True)
    stop_time = time.time()

    score = model.evaluate(X_test, Y_test, verbose=2)

    if not os.path.exists(model_path+"/HestonModels.txt"):
        with open(model_path+"/HestonModels.txt", "w") as output_file:
            pass

    # Appending test score to file
    with open(model_path+"/HestonModels.txt", "a") as output_file:
        output_file.write("\n")
        output_file.write(model_name+" has a score of: "+str(score)+", and took a total time of: "+str(stop_time - start_time))

    return 0    

X_train = np.loadtxt("Data/Sobol2_X_train.csv", delimiter = ",")
X_test = np.loadtxt("Data/Sobol2_X_test.csv", delimiter = ",")
Y_train = np.loadtxt("Data/Sobol2_Y_train.csv", delimiter = ",")
Y_test = np.loadtxt("Data/Sobol2_Y_test.csv", delimiter = ",")

X_train_price = np.loadtxt("Data/Sobol2_X_train_price.csv", delimiter = ",")
X_test_price = np.loadtxt("Data/Sobol2_X_test_price.csv", delimiter = ",")
Y_train_price = np.loadtxt("Data/Sobol2_Y_train_price.csv", delimiter = ",")
Y_test_price = np.loadtxt("Data/Sobol2_Y_test_price.csv", delimiter = ",")

X_train_single = np.loadtxt("Data/Sobol2_X_train_single.csv", delimiter = ",")
X_test_single = np.loadtxt("Data/Sobol2_X_test_single.csv", delimiter = ",")
Y_train_single = np.loadtxt("Data/Sobol2_Y_train_single.csv", delimiter = ",")
Y_test_single = np.loadtxt("Data/Sobol2_Y_test_single.csv", delimiter = ",")

Y_train_single = np.reshape(Y_train_single, (-1, 1))
Y_test_single = np.reshape(Y_test_single, (-1, 1))

X_train_grid = np.loadtxt("Data/Sobol2_X_train_grid.csv", delimiter = ",")
X_test_grid = np.loadtxt("Data/Sobol2_X_test_grid.csv", delimiter = ",")
Y_train_grid = np.loadtxt("Data/Sobol2_Y_train_grid.csv", delimiter = ",")
Y_test_grid = np.loadtxt("Data/Sobol2_Y_test_grid.csv", delimiter = ",")

X_train_wide = np.loadtxt("Data/Sobol2_X_train.csv", delimiter = ",")
X_test_wide = np.loadtxt("Data/Sobol2_X_test.csv", delimiter = ",")
Y_train_wide = np.loadtxt("Data/Sobol2_Y_train.csv", delimiter = ",")
Y_test_wide = np.loadtxt("Data/Sobol2_Y_test.csv", delimiter = ",")

data_set = [X_train, X_test, Y_train, Y_test]
data_set_price = [X_train_price, X_test_price, Y_train_price, Y_test_price]
data_set_single = [X_train_single, X_test_single, Y_train_single, Y_test_single]
data_set_grid = [X_train_grid, X_test_grid, Y_train_grid, Y_test_grid]
data_set_wide = [X_train_wide, X_test_wide, Y_train_wide, Y_test_wide]

### Using 200k sobol sets
price_imp_models_price = glob.glob("Models2/price_vs_imp/standard_price*.h5")
price_imp_models_imp = glob.glob("Models2/price_vs_imp/sobol*.h5")

price_imp_models_price_list = list(zip(price_imp_models_price, repeat(data_set_price)))
price_imp_models_imp_list = list(zip(price_imp_models_imp, repeat(data_set)))

standard_normal_models = glob.glob("Models2/standard_vs_normal/*.h5")
standard_normal_tanh_models = glob.glob("Models2/standard_vs_normal_tanh/*.h5")
standard_normal_mix_models = glob.glob("Models2/standard_vs_normal_mix/*.h5")

standard_normal_models_list = list(zip(standard_normal_models, repeat(data_set)))
standard_normal_tanh_models_list = list(zip(standard_normal_tanh_models, repeat(data_set)))
standard_normal_mix_models_list = list(zip(standard_normal_mix_models, repeat(data_set)))

### Using single sets, from 200k sobol
single_models = glob.glob("Models2/single/*.h5")

### Using grid set, 279936 sets
grid_models = glob.glob("Models2/grid_vs_sobol/standard*.h5")
sobol_grid_models = glob.glob("Models2/grid_vs_sobol/sobol*.h5")

single_list = list(zip(single_models, repeat(data_set_single)))
grid_list = list(zip(grid_models, repeat(data_set_grid)))
sobol_wide_list = list(zip(sobol_grid_models, repeat(data_set_wide)))

### Output scaling
output_scaling_models = glob.glob("Models2/output_scaling/*/*.h5")

output_scaling_models_list = list(zip(output_scaling_models, repeat(data_set)))

paral_list = [
    price_imp_models_price_list,
    price_imp_models_imp_list,
    standard_normal_models_list,
    standard_normal_tanh_models_list,
    grid_list,
    sobol_wide_list,
    output_scaling_models_list,
    single_list
]

name_list = [
    "price_imp_models_price_list",
    "price_imp_models_imp_list",
    "standard_normal_models_list",
    "standard_normal_tanh_models_list",
    "grid_list",
    "sobol_wide_list",
    "output_scaling_models_list",
    "single_list"
]

i = 0
for some_list in paral_list:
    print("Starting: ", name_list[i])
    cpu_cores = min(cpu_count(), len(some_list))
    pool = Pool(cpu_cores)
    res = pool.starmap(refit_model, some_list)
    pool.close()
    print(res)
    print("Stopping: ", name_list[i])
    i += 1

"""
norm_folder = "Models3/norms/"

### Grid data, Sobol
input_pre = np.loadtxt("Data/hestonSobolGridInput2_compare2_200000.csv", delimiter = ",")
output_pre = np.loadtxt("Data/hestonSobolGridImpVol2_compare2_200000.csv", delimiter = ",")
output_pre_price = np.loadtxt("Data/hestonSobolGridPrice2_compare2_200000.csv", delimiter = ",")

sobol_filter = np.all(output_pre != 0, axis = 1)
sobol_input = input_pre[sobol_filter, :]
sobol_output = output_pre[sobol_filter, :]

sobol_filter_price = np.all(output_pre != 0, axis = 1)
sobol_input_price = input_pre[sobol_filter_price, :]
sobol_output_price = output_pre_price[sobol_filter_price, :]

X_train, X_test, Y_train, Y_test = train_test_split(sobol_input, sobol_output, test_size=0.3, random_state=42)
X_train_price, X_test_price, Y_train_price, Y_test_price = train_test_split(sobol_input_price, sobol_output_price, test_size=0.3, random_state=42)

norm_feature_standard = MinMaxScaler()
standard_feature_standard = StandardScaler()
standard_label_standard = MinMaxScaler()

norm_feature_standard.fit(X_train)
standard_feature_standard.fit(X_train)
standard_label_standard.fit(Y_train)

joblib.dump(norm_feature_standard, norm_folder+"norm_features.pkl")
joblib.dump(standard_feature_standard, norm_folder+"standard_features.pkl")
joblib.dump(standard_label_standard, norm_folder+"norm_labels.pkl")

norm_feature_price = MinMaxScaler()

norm_feature_price.fit(X_train_price)

joblib.dump(norm_feature_price, norm_folder+"norm_feature_price.pkl")

np.savetxt("Data/Sobol2_X_train.csv", X_train, delimiter = ",")
np.savetxt("Data/Sobol2_X_test.csv", X_test, delimiter = ",")
np.savetxt("Data/Sobol2_Y_train.csv", Y_train, delimiter = ",")
np.savetxt("Data/Sobol2_Y_test.csv", Y_test, delimiter = ",")

np.savetxt("Data/Sobol2_X_train_price.csv", X_train_price, delimiter = ",")
np.savetxt("Data/Sobol2_X_test_price.csv", X_test_price, delimiter = ",")
np.savetxt("Data/Sobol2_Y_train_price.csv", Y_train_price, delimiter = ",")
np.savetxt("Data/Sobol2_Y_test_price.csv", Y_test_price, delimiter = ",")

### Grid data, grid
input_pre_grid = np.loadtxt("Data/hestonGridInput2_wide.csv", delimiter = ",")
output_pre_grid = np.loadtxt("Data/hestonGridImpVol2_wide.csv", delimiter = ",")

sobol_grid_filter = np.all(output_pre_grid != 0, axis = 1)
sobol_grid_input = input_pre_grid[sobol_grid_filter, :]
sobol_grid_output = output_pre_grid[sobol_grid_filter, :]

X_train_grid, X_test_grid, Y_train_grid, Y_test_grid = train_test_split(sobol_grid_input, sobol_grid_output, test_size=0.3, random_state=42)

norm_feature_grid = MinMaxScaler()

norm_feature_grid.fit(X_train_grid)

joblib.dump(norm_feature_grid, norm_folder+"norm_feature_grid.pkl")

np.savetxt("Data/Sobol2_X_train_grid.csv", X_train_grid, delimiter = ",")
np.savetxt("Data/Sobol2_X_test_grid.csv", X_test_grid, delimiter = ",")
np.savetxt("Data/Sobol2_Y_train_grid.csv", Y_train_grid, delimiter = ",")
np.savetxt("Data/Sobol2_Y_test_grid.csv", Y_test_grid, delimiter = ",")

### Big sobol for comparison
input_wide_pre = np.loadtxt("Data/hestonSobolGridInput2_compare_279936.csv", delimiter = ",")
output_wide_pre = np.loadtxt("Data/hestonSobolGridImpVol2_compare_279936.csv", delimiter = ",")

sobol_wide_filter = np.all(output_wide_pre != 0, axis = 1)
sobol_wide_input = input_wide_pre[sobol_wide_filter, :]
sobol_wide_output = output_wide_pre[sobol_wide_filter, :]

X_train_wide, X_test_wide, Y_train_wide, Y_test_wide = train_test_split(sobol_wide_input, sobol_wide_output, test_size=0.3, random_state=42)

norm_feature_wide = MinMaxScaler()

norm_feature_wide.fit(X_train_wide)

joblib.dump(norm_feature_wide, norm_folder+"norm_feature_wide.pkl")

np.savetxt("Data/Sobol2_X_train_wide.csv", X_train_wide, delimiter = ",")
np.savetxt("Data/Sobol2_X_test_wide.csv", X_test_wide, delimiter = ",")
np.savetxt("Data/Sobol2_Y_train_wide.csv", Y_train_wide, delimiter = ",")
np.savetxt("Data/Sobol2_Y_test_wide.csv", Y_test_wide, delimiter = ",")

### Single data
option_input = dg.option_input_generator()

total_comb = np.shape(input_pre)[0] * np.shape(option_input)[0]
total_cols = np.shape(input_pre)[1] + np.shape(option_input)[1]
total_options = np.shape(option_input)[0]
single_input = np.empty((total_comb, total_cols))
single_output = np.empty((total_comb, 1))
for i in range(np.shape(input_pre)[0]):
    for j in range(total_options):
        single_input[i*total_options+j, 0:np.shape(input_pre)[1]] = input_pre[i]
        single_input[i*total_options+j, (np.shape(input_pre)[1]) : total_cols] = option_input[j]
        single_output[i*total_options+j] = output_pre[i, j]
    
single_output = single_output.flatten()
single_output = np.reshape(single_output, (-1, 1))

sobol_single_filter = np.all(single_output != 0, axis = 1)
single_input = single_input[sobol_single_filter, :]
single_output = single_output[sobol_single_filter, :]

X_train_single, X_test_single, Y_train_single, Y_test_single = train_test_split(single_input, single_output, test_size=0.3, random_state=42)

norm_feature_single = MinMaxScaler()

norm_feature_single.fit(X_train_single)

joblib.dump(norm_feature_single, norm_folder+"norm_feature_single.pkl")

np.savetxt("Data/Sobol2_X_train_single.csv", X_train_single, delimiter = ",")
np.savetxt("Data/Sobol2_X_test_single.csv", X_test_single, delimiter = ",")
np.savetxt("Data/Sobol2_Y_train_single.csv", Y_train_single, delimiter = ",")
np.savetxt("Data/Sobol2_Y_test_single.csv", Y_test_single, delimiter = ",")

"""