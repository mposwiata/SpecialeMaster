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
from Thesis.Heston import DataGeneration as dg, ModelGenerator as mg
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

def NNModelNext(data_set : list, folder : str, model_name : str, n_layers : int, n_neurons : int, nn_type : str,  output_scaling : str, input_scaling : str) -> float:
    model_save = "Models5/"+folder+"/"+model_name+"_"+str(n_layers)+"_"+str(n_neurons)+".h5"
    model_path = "Models5/"+folder+"/"

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    X_train = data_set[0] 
    X_test = data_set[1]
    Y_train = data_set[2]
    Y_test = data_set[3]

    if input_scaling == "standardize":
        norm_features = StandardScaler()
        normal_in = True
    elif input_scaling == "normalize":
        norm_features = MinMaxScaler()
        normal_in = True
    else:
        normal_in = False
    

    if output_scaling == "standardize":
        norm_labels = StandardScaler()
        normal_out = True
    elif output_scaling == "normalize":
        norm_labels = MinMaxScaler()
        normal_out = True
    else:
        normal_out = False

    if normal_in:
        X_train = norm_features.fit_transform(X_train)
        X_test = norm_features.transform(X_test)
        ### saving feature normalization if it doesn't exists.
        joblib.dump(norm_features, model_path+"norm_feature.pkl")
    
    if normal_out:
        Y_train = norm_labels.fit_transform(Y_train)
        Y_test = norm_labels.transform(Y_test)
        joblib.dump(norm_labels, model_path+"norm_labels.pkl")

    if nn_type == "normal":
        model = nng.NN_generator(n_layers, n_neurons, np.shape(X_train)[1], np.shape(Y_train)[1])
    elif nn_type == "tanh":
        model = nng.NN_generator_tanh(n_layers, n_neurons, np.shape(X_train)[1], np.shape(Y_train)[1])
    elif nn_type == "mix":
        model = nng.NN_generator_mix(n_layers, n_neurons, np.shape(X_train)[1], np.shape(Y_train)[1])
    elif nn_type == "regularization":
        model = nng.NN_generator_regul(n_layers, n_neurons, np.shape(X_train)[1], np.shape(Y_train)[1])
    elif nn_type == "dropput":
        model = nng.NN_generator_dropout(n_layers, n_neurons, np.shape(X_train)[1], np.shape(Y_train)[1])
    else:
        model = nng.NN_generator_mix_noise(n_layers, n_neurons, np.shape(X_train)[1], np.shape(Y_train)[1])

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
        print("overfit, saving overfit model")
        model.save(model_save)

    if not os.path.exists(model_path+"/HestonModels.txt"):
        with open(model_path+"/HestonModels.txt", "w") as output_file:
            pass

    # Appending test score to file
    with open(model_path+"/HestonModels.txt", "a") as output_file:
        output_file.write("\n")
        output_file.write(model_save+" has a score of: "+str(score)+", and took a total time of: "+str(stop_time - start_time))

    print("Done with: ", model_save)
    return score

def NN_mc_model_1(data_set : list, folder : str, model_name : str, n_layers : int, n_neurons : int, nn_type : str,  output_scaling : str, input_scaling : str, include_zero : bool, special_type : str = None) -> float:
    X_train = data_set[0] 
    X_test = data_set[1]
    Y_train = data_set[2]
    Y_test = data_set[3]

    if special_type == "mat":
        X_train, Y_train = mg.transform_mat(X_train, Y_train)
        X_test, Y_test = mg.transform_mat(X_test, Y_test)
    elif special_type == "single":
        X_train, Y_train = mg.transform_single(X_train, Y_train)
        X_test, Y_test = mg.transform_single(X_test, Y_test)

    if include_zero:
        train_index = np.all(Y_train != -1, axis = 1)
        test_index = np.all(Y_test != -1, axis = 1)
    else:
        train_index = np.all(Y_train > 0, axis = 1)
        test_index = np.all(Y_test > 0, axis = 1)
    X_train = X_train[train_index, :]
    Y_train = Y_train[train_index, :]
    X_test = X_test[test_index, :]
    Y_test = Y_test[test_index, :]
    
    try:
        data_set = [X_train, X_test, Y_train, Y_test]
        score = NNModelNext(data_set, folder, model_name, n_layers, n_neurons, nn_type,  output_scaling, input_scaling)

        return score
    except:
        return 0
    
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

if __name__ == '__main__':
    print("nothing")