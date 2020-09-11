import numpy as np
import pandas as pd
#import keras
import tensorflow
import matplotlib.pyplot as plt
from multiprocess import Pool
from ANN_black import model_bs as mbs
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Flatten
from keras import backend
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.optimizers import schedules, Adam

train_input = np.loadtxt("train_input_bs.csv", delimiter=",")
train_output = np.loadtxt("train_output_bs.csv", delimiter=",")
test_input = np.loadtxt("test_input_bs.csv", delimiter=",")
test_output = np.loadtxt("test_output_bs.csv", delimiter=",")

pool = Pool(4)
input_set = [
    [train_input, train_output, test_input, test_output, 1000, 128, 0.01, "test7.h5", 500, 0.9, 0.1],
    [train_input, train_output, test_input, test_output, 1000, 128, 0.01, "test7.h5", 1000, 0.9, 0.1],
    [train_input, train_output, test_input, test_output, 1000, 128, 0.01, "test7.h5", 2500, 0.9, 0.1],
    [train_input, train_output, test_input, test_output, 1000, 128, 0.01, "test7.h5", 5000, 0.9, 0.1]
]
res = pool.starmap(mbs.model_train, input_set)
print(res)


"""
# Number of nodes
nodes = 1000

# Model creation
model = Sequential()

# Layer 1
model.add(Dense(nodes, input_shape=(4,)))
model.add(Activation('softplus')) # Rectified Linear Unit, f(x) = max(x,0)
model.add(Dropout(0.1))

# Layer 2, output
model.add(Dense(1))
model.add(Activation('elu'))

lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9
)

opt = Adam(learning_rate=lr_schedule)

model.compile(
    loss = 'mse', #mean squared error
    optimizer = opt
)

callbacks_list = [EarlyStopping(monitor='val_loss', patience=3)] #stop if cross validation fails to decrease over #patience epochs

model.fit(train_input, train_output, epochs=30, batch_size=128, validation_split=0.2, verbose=1, callbacks=callbacks_list, use_multiprocessing=True)

model.evaluate(test_input, test_output, verbose = 2)

model.save('bs_model.h5')
"""