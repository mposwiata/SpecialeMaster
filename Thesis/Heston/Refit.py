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

"""
input_pre = np.loadtxt("Data/hestonSobolGridInput2_compare2_200000.csv", delimiter = ",")
output_pre = np.loadtxt("Data/hestonSobolGridImpVol2_compare2_200000.csv", delimiter = ",")

sobol_filter = np.all(output_pre != 0, axis = 1)
sobol_input = input_pre[sobol_filter, :]
sobol_output = output_pre[sobol_filter, :]

X_train, X_test, Y_train, Y_test = train_test_split(sobol_input, sobol_output, test_size=0.3, random_state=42)

np.savetxt("Data/Sobol2_X_train.csv", X_train, delimiter = ",")
np.savetxt("Data/Sobol2_X_test.csv", X_test, delimiter = ",")
np.savetxt("Data/Sobol2_Y_train.csv", Y_train, delimiter = ",")
np.savetxt("Data/Sobol2_Y_test.csv", Y_test, delimiter = ",")
"""

X_train = np.loadtxt("Data/Sobol2_X_train.csv", delimiter = ",")
X_test = np.loadtxt("Data/Sobol2_X_test.csv", delimiter = ",")
Y_train = np.loadtxt("Data/Sobol2_Y_train.csv", delimiter = ",")
Y_test = np.loadtxt("Data/Sobol2_Y_test.csv", delimiter = ",")

data_set = [X_train, X_test, Y_train, Y_test]

price_imp_models = glob.glob("Models2/price_vs_imp/*.h5")
standard_normal_models = glob.glob("Models2/stardard_vs_normal/*.h5")
standard_normal_tanh_models = glob.glob("Models2/stardard_vs_normal_tanh/*.h5")
standard_normal_mix_models = glob.glob("Models2/stardard_vs_normal_mix/*.h5")

model_list = price_imp_models + standard_normal_models + standard_normal_tanh_models + standard_normal_mix_models

cpu_cores = min(cpu_count(), len(model_list))
# parallel
pool = Pool(cpu_cores)
res_models = pool.starmap(refit_model, zip(model_list, repeat(data_set)))
pool.close()
print(res_models)