import numpy as np
import tensorflow as tf 
import joblib
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import load_model
import os

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

    if score > 0.7: #if overfitting, save that model
        model.save(model_save)

    if not os.path.exists(model_path+"/HestonModels.txt"):
        with open(model_path+"/HestonModels.txt", "w") as output_file:
            pass

    # Appending test score to file
    with open(model_path+"/HestonModels.txt", "a") as output_file:
        output_file.write("\n")
        output_file.write(model_name+" has a score of: "+str(score)+", and took a total time of: "+str(stop_time - start_time))

    return score