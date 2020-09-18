import numpy as np
import pandas as pd
#import keras
import tensorflow
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Flatten
from keras import backend
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.optimizers import schedules, Adam

def model_train(train_input, train_output, test_input, test_output, 
    nodes, batch_size, learning_rate, file_name,
    decay_steps, decay_rate, validation_split):
    # Number of nodes
    nodes = nodes

    # Model creation
    model = Sequential()

    # Layer 1
    model.add(Dense(nodes, input_shape=(4,)))
    model.add(Activation('softplus')) # Rectified Linear Unit, f(x) = max(x,0)
    model.add(Dropout(0.1))

    # Layer 2
    model.add(Dense(nodes))
    model.add(Activation('softplus')) # Rectified Linear Unit, f(x) = max(x,0)
    model.add(Dropout(0.1))

    # Layer 3
    model.add(Dense(nodes))
    model.add(Activation('softplus')) # Rectified Linear Unit, f(x) = max(x,0)
    model.add(Dropout(0.1))

    # Layer 4, output
    model.add(Dense(1))
    model.add(Activation('elu'))

    lr_schedule = schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )

    opt = Adam(learning_rate=lr_schedule)

    model.compile(
        loss = 'mse', #mean squared error
        optimizer = opt
    )

    callbacks_list = [EarlyStopping(monitor='val_loss', patience=15)] #stop if cross validation fails to decrease over #patience epochs

    model.fit(train_input, train_output, epochs=30, batch_size=batch_size, validation_split=validation_split, verbose=1, callbacks=callbacks_list)

    model.save(file_name)

    return model.evaluate(test_input, test_output, verbose = 2)

    