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

train_input = np.loadtxt("train_input_sabr.csv", delimiter=",")
train_output = np.loadtxt("train_output_sabr_approx.csv", delimiter=",")
test_input = np.loadtxt("test_input_sabr.csv", delimiter=",")
test_output = np.loadtxt("test_output_sabr_approx.csv", delimiter=",")

# Number of nodes
nodes = 1000

# Model creation
model = Sequential()

# Layer 1
model.add(Dense(nodes, input_shape=(14,)))
model.add(Activation('softplus')) # Rectified Linear Unit, f(x) = max(x,0)
model.add(Dropout(0.25))

# Layer 2, output
model.add(Dense(10))
model.add(Activation('elu'))

model.compile(
    loss = 'mse', #mean squared error
    optimizer = 'adam'
)

callbacks_list = [EarlyStopping(monitor='val_loss', patience=2)] #stop if cross validation fails to decrease

model.fit(train_input, train_output, epochs=50, batch_size=250, validation_split=0.1, verbose=2, callbacks=callbacks_list)

model.save('sabr_approx_model.h5')

model.evaluate(test_input, test_output, verbose = 2)