import numpy as np
import pandas as pd
#import keras
import tensorflow
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization
from keras import backend
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.optimizers import schedules, Adam

def model_train(train_input, train_output, test_input, test_output, 
    nodes, batch_size, learning_rate, file_name,
    decay_steps, decay_rate, validation_split, 
    dropout, batch_normalization, biased, patience):
    # Number of nodes
    nodes = nodes

    # Model creation
    model = Sequential()

    # Normalization layer / layer 1
    if batch_normalization:
        model.add(BatchNormalization(input_shape=(14,))) 

        model.add(Dense(nodes, use_bias=biased, kernel_initializer='random_normal', bias_initializer='zeros'))
        model.add(Activation('softplus'))
        model.add(Dropout(dropout))
    else:
        model.add(Dense(nodes, input_shape=(14,), use_bias=False, kernel_initializer='random_normal', bias_initializer='zeros'))
        model.add(Activation('softplus'))
        model.add(Dropout(dropout))
    
    # Layer 2, output
    model.add(Dense(10))
    #model.add(Activation('elu'))

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

    callbacks_list = [EarlyStopping(monitor='val_loss', patience=patience)] #stop if cross validation fails to decrease over #patience epochs

    model.fit(train_input, train_output, epochs=100, batch_size=batch_size, 
    validation_data=(test_input, test_output), verbose=0, callbacks=callbacks_list, shuffle=True)

    model.save(file_name)

    return 0

    