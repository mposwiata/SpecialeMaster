import numpy as np
import joblib
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.preprocessing import StandardScaler
from keras import backend as k
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

input1 = np.loadtxt("Data/hestonSingleInput.csv", delimiter=",")
output1 = np.loadtxt("Data/hestonSingleOutput.csv", delimiter=",")

# filtering out 0 and 0.1 values
filterArray = np.nonzero((output1 != 0) & (output1 != 0.1))
input1 = input1[filterArray]
output1 = output1[filterArray]

output_data = np.reshape(output1, (-1, 1))

X_train, X_test, y_train, y_test = train_test_split(input1, output_data, test_size=0.3, random_state=42)

norm_features = StandardScaler() #MinMaxScaler(feature_range = (-1, 1))
norm_labels = StandardScaler()

X_train_norm = norm_features.fit_transform(X_train)
Y_train_norm = norm_labels.fit_transform(y_train)

X_test_norm = norm_features.transform(X_test)
Y_test_norm = norm_labels.transform(y_test)

model = nng.NNGenerator(4, 1000, np.shape(input1)[1], np.shape(output_data)[1])

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
    saveString = "Models/HestonSinglePrice/Heston_imp_single_"+str(i)+".h5"
    no = i
    if os.path.isfile(saveString) == False:
        break

# Saving model
model.save("Models/HestonSinglePrice/Heston_imp_single_"+str(no)+".h5")

# Saving normalization parameters
joblib.dump(norm_features, "Models/HestonSinglePrice/norm_features_"+str(no)+".pkl")
joblib.dump(norm_labels, "Models/HestonSinglePrice/norm_labels_"+str(no)+".pkl")