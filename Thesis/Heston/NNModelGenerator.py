import numpy as np
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

def NNModel(inputArray : np.ndarray, outputArray : np.ndarray, nLayers : int, nNeurons : int, modelname : str):
    X_train, X_test, y_train, y_test = train_test_split(inputArray, outputArray, test_size=0.3, random_state=42)

    norm_features = StandardScaler() # MinMaxScaler(feature_range = (-1, 1))
    norm_labels = StandardScaler()

    X_train_norm = norm_features.fit_transform(X_train)
    Y_train_norm = norm_labels.fit_transform(y_train)

    X_test_norm = norm_features.transform(X_test)
    Y_test_norm = norm_labels.transform(y_test)

    model = nng.NNGenerator(nLayers, nNeurons, np.shape(inputArray)[1], np.shape(outputArray)[1])

    adam = Adam(lr = 0.01)

    model.compile(
        loss = 'mean_squared_error', #mean squared error
        optimizer = adam
        )

    callbacks_list = [
        LearningRateScheduler(lr_schedule, verbose = 0),
        EarlyStopping(monitor='val_loss', patience=15)
    ]

    model.fit(X_train_norm, Y_train_norm, epochs=100, batch_size=256, verbose = 0, callbacks = callbacks_list, validation_split = 0.1, shuffle=True)
    
    score = model.evaluate(X_test_norm, Y_test_norm, verbose=2)
    print(modelname+" has a testscore of: "+str(score))

    no = 0
    for i in range(1,100):
        saveString = "Models/Heston/"+modelname+"_"+str(i)+".h5"
        no = i
        if os.path.isfile(saveString) == False:
            break

    # Saving model
    model.save("Models/Heston/"+modelname+"_"+str(no)+".h5")

    # Saving normalization parameters
    joblib.dump(norm_features, "Models/Heston/"+modelname+"_norm_features_"+str(no)+".pkl")
    joblib.dump(norm_labels, "Models/Heston/"+modelname+"_norm_labels_"+str(no)+".pkl")

    with open("Models/Heston/HestonModels.txt", "w") as output_file:
        output_file.write(modelname+" has a score of: "+str(score))