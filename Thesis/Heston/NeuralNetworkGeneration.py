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

input1 = np.loadtxt("Data/hestonPriceGridInput.csv", delimiter=",")
output1 = np.loadtxt("Data/hestonPriceGridOutput.csv", delimiter=",")

X_train, X_test, y_train, y_test = train_test_split(input1, output1, test_size=0.3, random_state=42)

norm_features = MinMaxScaler(feature_range = (-1, 1)) #StandardScaler()
norm_labels = StandardScaler()

X_train_norm = norm_features.fit_transform(X_train)
Y_train_norm = norm_labels.fit_transform(y_train)

X_test_norm = norm_features.transform(X_test)
Y_test_norm = norm_labels.transform(y_test)

model = nng.NNGenerator(4, 1000, np.shape(input1)[1], np.shape(output1)[1])

adam = Adam(lr = 0.1)

model.compile(
    loss = 'mean_squared_error', #mean squared error
    optimizer = adam
    )

callbacks_list = [
    LearningRateScheduler(lr_schedule, verbose = 0),
    EarlyStopping(monitor='val_loss', patience=25)
]

model.fit(X_train_norm, Y_train_norm, epochs=100, batch_size=1024, verbose = 2, callbacks = callbacks_list, validation_split = 0.1)

score=model.evaluate(X_test_norm, Y_test_norm, verbose=2)

print(score)

# Saving model
model.save("Models/HestonGridPrice/Heston_price_grid_1.h5")

# Saving normalization parameters
joblib.dump(norm_features, "Models/HestonGridPrice/norm_features_1.pkl")
joblib.dump(norm_labels, "Models/HestonGridPrice/norm_labels_1.pkl")