import numpy as np
import joblib
import itertools
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import backend as k
import sys
import os
sys.path.append(os.getcwd()) # added for calc server support

from Thesis import NeuralNetworkGenerator as nng
from Thesis.Heston import DataGeneration as dg

def lr_schedule(n, alpha):
    a, b = 1e-4, 1e-2
    n1, n2, n3 = 0, 24, 74

    if n <= n2:
        return (a - b)/(n1 - n2) * n - (a*n2 - b*n1) / (n1 - n2)
    elif n2 < n < n3:
        return -(a - b)/(n2 - n3) * n + (a*n2 - b*n3) / (n2 - n3)
    else:
        return a

def singleNN(inputArray : np.ndarray, outputArray : np.ndarray, nLayers : int, nNeurons : int, modelname : str):
    X_train, X_test, y_train, y_test = train_test_split(input2, output_data, test_size=0.3, random_state=42)

    norm_features = StandardScaler() #MinMaxScaler(feature_range = (-1, 1))
    norm_labels = StandardScaler()

    X_train_norm = norm_features.fit_transform(X_train)
    Y_train_norm = norm_labels.fit_transform(y_train)

    X_test_norm = norm_features.transform(X_test)
    Y_test_norm = norm_labels.transform(y_test)

    model = nng.NNGenerator(4, 1000, np.shape(input2)[1], np.shape(output_data)[1])

    adam = Adam(lr = 0.01)

    model.compile(
        loss = 'mean_squared_error', #mean squared error
        optimizer = adam
        )

    callbacks_list = [
        LearningRateScheduler(lr_schedule, verbose = 0),
        EarlyStopping(monitor='val_loss', patience=15)
    ]

    model.fit(X_train_norm, Y_train_norm, epochs=100, batch_size=1024, verbose = 2, callbacks = callbacks_list, validation_split = 0.1)

    score=model.evaluate(X_test_norm, Y_test_norm, verbose=2)

    print(score)

    no = 0
    for i in range(1, 100):
        saveString = "Models/HestonSinglePrice/Heston_price_single_"+str(i)+".h5"
        no = i
        if os.path.isfile(saveString) == False:
            break

    # Saving model
    model.save("Models/HestonSinglePrice/Heston_price_single_"+str(no)+".h5")

    # Saving normalization parameters
    joblib.dump(norm_features, "Models/HestonSinglePrice/norm_features_price_"+str(no)+".pkl")
    joblib.dump(norm_labels, "Models/HestonSinglePrice/norm_labels_price_"+str(no)+".pkl")

model_input = dg.modelInputGenerator
option_input = dg.optionInputGenerator

output1 = np.loadtxt("Data/hestonPriceGridOutput.csv", delimiter=",")
# generating data for nn with all inputs and 1 output price
total_comb = np.shape(model_input)[0] * np.shape(output1)[1]
total_cols = np.shape(model_input)[1] + np.shape(option_input)[1]
total_options = np.shape(option_input)[0]
input2 = np.empty((total_comb, total_cols))
output2 = np.empty((total_comb, 1))
for i in range(np.shape(model_input)[0]):
    for j in range(total_options):
        input2[i*total_options+j, 0:np.shape(model_input)[1]] = model_input[i]
        input2[i*total_options+j, (np.shape(model_input)[1]) : total_cols] = option_input[j]
        output2[i*total_options+j] = output1[i, j]

# filtering out 0 values
output2 = output2.flatten()
filterArray = np.nonzero((output2 != 0))
print(np.shape(input2))
input2 = input2[filterArray]
print(np.shape(input2))
output2 = output2[filterArray]
print(np.shape(output2))

output_data = np.reshape(output2, (-1, 1))

X_train, X_test, y_train, y_test = train_test_split(input2, output_data, test_size=0.3, random_state=42)

norm_features = StandardScaler() #MinMaxScaler(feature_range = (-1, 1))
norm_labels = StandardScaler()

X_train_norm = norm_features.fit_transform(X_train)
Y_train_norm = norm_labels.fit_transform(y_train)

X_test_norm = norm_features.transform(X_test)
Y_test_norm = norm_labels.transform(y_test)

model = nng.NNGenerator(4, 1000, np.shape(input2)[1], np.shape(output_data)[1])

adam = Adam(lr = 0.1)

model.compile(
    loss = 'mean_squared_error', #mean squared error
    optimizer = adam
    )

callbacks_list = [
    LearningRateScheduler(lr_schedule, verbose = 0),
    EarlyStopping(monitor='val_loss', patience=15)
]

model.fit(X_train_norm, Y_train_norm, epochs=100, batch_size=1024, verbose = 2, callbacks = callbacks_list, validation_split = 0.1)

score=model.evaluate(X_test_norm, Y_test_norm, verbose=2)

print(score)

no = 0
for i in range(1, 100):
    saveString = "Models/HestonSinglePrice/Heston_price_single_"+str(i)+".h5"
    no = i
    if os.path.isfile(saveString) == False:
        break

# Saving model
model.save("Models/HestonSinglePrice/Heston_price_single_"+str(no)+".h5")

# Saving normalization parameters
joblib.dump(norm_features, "Models/HestonSinglePrice/norm_features_price_"+str(no)+".pkl")
joblib.dump(norm_labels, "Models/HestonSinglePrice/norm_labels_price_"+str(no)+".pkl")
