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
        saveString = "Models/HestonTahn/"+model_name+"_"+str(i)+".h5"
        no = i
        if os.path.isfile(saveString) == False:
            break

    # Saving model
    model.save("Models/HestonTahn/"+model_name+"_"+str(no)+".h5")

    # Saving normalization parameters
    joblib.dump(norm_features, "Models/HestonTahn/"+model_name+"_norm_features_"+str(no)+".pkl")
    joblib.dump(norm_labels, "Models/HestonTahn/"+model_name+"_norm_labels_"+str(no)+".pkl")

    # Appending test score to file
    with open("Models/HestonTahn/HestonModels.txt", "a") as output_file:
        output_file.write("\n")
        output_file.write(model_name+" has a score of: "+str(score)+", and took a total time of: "+str(stop_time - start_time))

    print("Stopping: "+model_name)
    return score

def NNModel(input_array : np.ndarray, output_array : np.ndarray, n_layers : int, n_neurons : int, model_name : str) -> float:
    print("Starting: "+model_name)
    X_train, X_test, y_train, y_test = train_test_split(input_array, output_array, test_size=0.3, random_state=42)

    norm_features = StandardScaler() # MinMaxScaler(feature_range = (-1, 1))
    norm_labels = StandardScaler()

    X_train_norm = norm_features.fit_transform(X_train)
    Y_train_norm = norm_labels.fit_transform(y_train)

    X_test_norm = norm_features.transform(X_test)
    Y_test_norm = norm_labels.transform(y_test)

    model = nng.NN_generator(n_layers, n_neurons, np.shape(input_array)[1], np.shape(output_array)[1])

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
        saveString = "Models/Heston/"+model_name+"_"+str(i)+".h5"
        no = i
        if os.path.isfile(saveString) == False:
            break

    # Saving model
    model.save("Models/Heston/"+model_name+"_"+str(no)+".h5")

    # Saving normalization parameters
    joblib.dump(norm_features, "Models/Heston/"+model_name+"_norm_features_"+str(no)+".pkl")
    joblib.dump(norm_labels, "Models/Heston/"+model_name+"_norm_labels_"+str(no)+".pkl")

    # Appending test score to file
    with open("Models/Heston/HestonModels.txt", "a") as output_file:
        output_file.write("\n")
        output_file.write(model_name+" has a score of: "+str(score)+", and took a total time of: "+str(stop_time - start_time))

    print("Stopping: "+model_name)
    return score