import numpy as np
import pandas as pd
import keras
import tensorflow
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Flatten
from keras import backend
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

input_data = np.loadtxt("input.csv", delimiter=",")
output_data = np.loadtxt("output.csv")
input_data[:,0] = input_data[:,0] / input_data[:,1] #normalize, spot / strike
output_data = output_data / input_data[:,1] #normalize, price / strike
input_data[:,1] = 1 #set strike = 1

x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size = 0.2, random_state = 42)

# Number of nodes
nodes = 1000

# Model creation
model = Sequential()

# Layer 1
model.add(Dense(nodes, input_shape=(4,)))
model.add(Activation('relu')) # Rectified Linear Unit, f(x) = max(x,0)
model.add(Dropout(0.25))

"""
# Layer 2
model.add(Dense(nodes))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Layer 3
model.add(Dense(nodes))
model.add(Activation('relu'))
model.add(Dropout(0.25))
"""

# Layer 4, output
model.add(Dense(1))
model.add(Activation('elu'))

model.compile(
    loss = 'mse', #mean squared error
    optimizer = 'rmsprop'
)

callbacks_list = [EarlyStopping(monitor='val_loss', patience=1)] #stop if cross validation fails to decrease

model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=2, callbacks=callbacks_list)

model.save('first_model.h5')

model.evaluate(x_test, y_test, verbose = 2)

predictions = model.predict(x_test)
plt.figure(figsize = (10,10))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.plot([0,1], [0,1], 'r')
plt.grid(True)
plt.savefig("plot1.jpeg")