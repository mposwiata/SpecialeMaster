import numpy as np
import time
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras import backend as k
import joblib
import sys
import os
sys.path.append(os.getcwd()) # added for calc server support

from Thesis import NeuralNetworkGenerator as nng
from sklearn.model_selection import train_test_split

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
    

def NNModelTanh(input_array : np.ndarray, output_array : np.ndarray, n_layers : int, n_neurons : int, model_name : str) -> float:
    print("Starting: "+model_name)
    X_train, X_test, y_train, y_test = train_test_split(input_array, output_array, test_size=0.3, random_state=42)

    norm_features = StandardScaler() # MinMaxScaler(feature_range = (-1, 1))
    norm_labels = StandardScaler()

    X_train_norm = norm_features.fit_transform(X_train)
    Y_train_norm = norm_labels.fit_transform(y_train)

    X_test_norm = norm_features.transform(X_test)
    Y_test_norm = norm_labels.transform(y_test)

    model = nng.NN_generator_tanh(n_layers, n_neurons, np.shape(input_array)[1], np.shape(output_array)[1])

    adam = Adam(lr = 0.01)

    model.compile(
        loss = 'mean_squared_error', #mean squared error
        optimizer = adam
        )

    callbacks_list = [
        LearningRateScheduler(lr_schedule, verbose = 0),
        EarlyStopping(monitor='val_loss', patience=25)
    ]

    start_time = time.time()
    model.fit(X_train_norm, Y_train_norm, epochs=100, batch_size=256, verbose = 0, callbacks = callbacks_list, validation_split = 0.1, shuffle=True)
    stop_time = time.time()

    score = model.evaluate(X_test_norm, Y_test_norm, verbose=2)

    # checking file name
    no = 0
    for i in range(1,100):
        saveString = "Models/HestonTanh/"+model_name+"_"+str(i)+".h5"
        no = i
        if os.path.isfile(saveString) == False:
            break

    # Saving model
    model.save("Models/HestonTanh/"+model_name+"_"+str(no)+".h5")

    # Saving normalization parameters
    joblib.dump(norm_features, "Models/HestonTanh/"+model_name+"_norm_features_"+str(no)+".pkl")
    joblib.dump(norm_labels, "Models/HestonTanh/"+model_name+"_norm_labels_"+str(no)+".pkl")

    # Appending test score to file
    with open("Models/HestonTanh/HestonModels.txt", "a") as output_file:
        output_file.write("\n")
        output_file.write(model_name+" has a score of: "+str(score)+", and took a total time of: "+str(stop_time - start_time))

    print("Stopping: "+model_name)
    return score

def NNModel(input_array : np.ndarray, output_array : np.ndarray, n_layers : int, n_neurons : int, model_name : str, normal_out : bool = True, nn_type : str = "normal", scalar : str = "stardardize") -> float:
    print("Starting: "+model_name)
    if normal_out:
        if nn_type == "normal":
            folder_name = "Heston"
        elif nn_type == "tanh":
            folder_name = "HestonTanh"
        else:
            folder_name = "HestonMix"
    else:
        if nn_type == "normal":
            folder_name = "Heston_non_normal"
        elif nn_type == "tanh":
            folder_name = "Heston_non_normal_tanh"
        else:
            folder_name = "Heston_non_normal_mix"
    
    if not os.path.exists("Models2/"+folder_name):
        os.makedirs("Models2/"+folder_name)

    # checking file name
    no = 0
    for i in range(1,100):
        saveString = "Models2/"+folder_name+"/"+model_name+"_"+str(i)+".h5"
        no = i
        if os.path.isfile(saveString) == False:
            break
    
    model_path = "Models2/"+folder_name+"/"+model_name+"_"+str(no)+".h5"

    X_train, X_test, Y_train, Y_test = train_test_split(input_array, output_array, test_size=0.3, random_state=42)

    if scalar == "stardardize":
        norm_features = StandardScaler()
    else:
        norm_features = MinMaxScaler()

    if normal_out:
        norm_labels = StandardScaler()
        Y_train = norm_labels.fit_transform(Y_train)
        Y_test = norm_labels.transform(Y_test)

    X_train_norm = norm_features.fit_transform(X_train)
    
    X_test_norm = norm_features.transform(X_test)
    
    if nn_type == "normal":
        model = nng.NN_generator(n_layers, n_neurons, np.shape(input_array)[1], np.shape(output_array)[1])
    elif nn_type == "tanh":
        model = nng.NN_generator_tanh(n_layers, n_neurons, np.shape(input_array)[1], np.shape(output_array)[1])
    else:
        model = nng.NN_generator_mix(n_layers, n_neurons, np.shape(input_array)[1], np.shape(output_array)[1])

    adam = Adam()

    model.compile(
        loss = 'mean_squared_error', #mean squared error
        optimizer = adam
        )

    callbacks_list = [
        LearningRateScheduler(lr_schedule, verbose = 0),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True)
    ]

    start_time = time.time()
    model.fit(X_train_norm, Y_train, epochs=100, batch_size=1024, verbose = 0, callbacks = callbacks_list, validation_split = 0.1, shuffle=True)
    stop_time = time.time()

    score = model.evaluate(X_test_norm, Y_test, verbose=2)

    # Saving model
    model.save(model_path)

    # Saving normalization parameters
    joblib.dump(norm_features, "Models2/"+folder_name+"/"+model_name+"_norm_features_"+str(no)+".pkl")
    
    if normal_out:
        joblib.dump(norm_labels, "Models2/"+folder_name+"/"+model_name+"_norm_labels_"+str(no)+".pkl")

    # Appending test score to file
    with open("Models2/"+folder_name+"/HestonModels.txt", "a") as output_file:
        output_file.write("\n")
        output_file.write(model_name+" has a score of: "+str(score)+", and took a total time of: "+str(stop_time - start_time))

    print("Stopping: "+model_name)
    return score

def NNModelMix(input_array : np.ndarray, output_array : np.ndarray, n_layers : int, n_neurons : int, model_name : str, normal_out : bool = True) -> float:
    print("Starting: "+model_name)
    X_train, X_test, y_train, y_test = train_test_split(input_array, output_array, test_size=0.3, random_state=42)

    norm_features = StandardScaler() # MinMaxScaler(feature_range = (-1, 1))
    norm_labels = StandardScaler()

    X_train_norm = norm_features.fit_transform(X_train)
    Y_train_norm = norm_labels.fit_transform(y_train)

    X_test_norm = norm_features.transform(X_test)
    Y_test_norm = norm_labels.transform(y_test)

    model = nng.NN_generator_mix(n_layers, n_neurons, np.shape(input_array)[1], np.shape(output_array)[1])

    adam = Adam(lr = 0.01)

    model.compile(
        loss = 'mean_squared_error', #mean squared error
        optimizer = adam
        )

    callbacks_list = [
        LearningRateScheduler(lr_schedule, verbose = 0),
        EarlyStopping(monitor='val_loss', patience=15)
    ]

    start_time = time.time()
    model.fit(X_train_norm, Y_train_norm, epochs=100, batch_size=256, verbose = 0, callbacks = callbacks_list, validation_split = 0.1, shuffle=True)
    stop_time = time.time()

    score = model.evaluate(X_test_norm, Y_test_norm, verbose=2)

    # checking file name
    no = 0
    for i in range(1,100):
        saveString = "Models/HestonMix/"+model_name+"_"+str(i)+".h5"
        no = i
        if os.path.isfile(saveString) == False:
            break

    # Saving model
    model.save("Models/HestonMix/"+model_name+"_"+str(no)+".h5")

    # Saving normalization parameters
    joblib.dump(norm_features, "Models/HestonMix/"+model_name+"_norm_features_"+str(no)+".pkl")
    joblib.dump(norm_labels, "Models/HestonMix/"+model_name+"_norm_labels_"+str(no)+".pkl")

    # Appending test score to file
    with open("Models/HestonMix/HestonModels.txt", "a") as output_file:
        output_file.write("\n")
        output_file.write(model_name+" has a score of: "+str(score)+", and took a total time of: "+str(stop_time - start_time))

    print("Stopping: "+model_name)
    return score