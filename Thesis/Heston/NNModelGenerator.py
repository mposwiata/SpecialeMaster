import numpy as np
import time
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras import backend as k
import joblib
import sys
import os
sys.path.append(os.getcwd()) # added for calc server support

from Thesis import NeuralNetworkGenerator as nng
from sklearn.model_selection import train_test_split

def lr_schedule(n, alpha):
    a, b = 1e-4, 1e-2
    n1, n2, n3 = 0, 24, 74

    if n <= n2:
        return (a - b)/(n1 - n2) * n - (a*n2 - b*n1) / (n1 - n2)
    elif n2 < n < n3:
        return -(a - b)/(n2 - n3) * n + (a*n2 - b*n3) / (n2 - n3)
    else:
        return a

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

def NNModel(input_array : np.ndarray, output_array : np.ndarray, n_layers : int, n_neurons : int, model_name : str, normal_out : bool = True, nn_type : str = "normal") -> float:
    print("Starting: "+model_name)
    X_train, X_test, Y_train, Y_test = train_test_split(input_array, output_array, test_size=0.3, random_state=42)

    norm_features = StandardScaler() # MinMaxScaler(feature_range = (-1, 1))
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
    model.fit(X_train_norm, Y_train, epochs=100, batch_size=256, verbose = 0, callbacks = callbacks_list, validation_split = 0.1, shuffle=True)
    stop_time = time.time()

    score = model.evaluate(X_test_norm, Y_test, verbose=2)

    # checking file name
    no = 0
    for i in range(1,100):
        saveString = "Models/Heston/"+model_name+"_"+str(i)+".h5"
        no = i
        if os.path.isfile(saveString) == False:
            break

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
    
    if not os.path.exists("Models/"+folder_name):
        os.makedir("Models/"+folder_name)
    # Saving model
    model.save("Models/"+folder_name+"/"+model_name+"_"+str(no)+".h5")

    # Saving normalization parameters
    joblib.dump(norm_features, "Models/"+folder_name+"/"+model_name+"_norm_features_"+str(no)+".pkl")
    
    if normal_out:
        joblib.dump(norm_labels, "Models/"+folder_name+"/"+model_name+"_norm_labels_"+str(no)+".pkl")

    # Appending test score to file
    with open("Models/"+folder_name+"/HestonModels.txt", "a") as output_file:
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