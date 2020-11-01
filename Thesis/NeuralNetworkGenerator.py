import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.initializers import VarianceScaling

def NN_generator(n_layers : int, n_nodes : int, input_dim : int, output_dim : int) -> Sequential:
    model = Sequential()

    # Layer 1, input
    layer1 = Dense(n_nodes, activation = 'softplus',
        kernel_initializer=VarianceScaling(), 
        input_shape = (input_dim,))
    model.add(layer1)
    
    # Hidden layers
    if n_layers > 1:
        for i in range(n_layers - 1):
            layer = Dense(n_nodes, activation = 'softplus',
                kernel_initializer=VarianceScaling())
            model.add(layer)
    
    # Output layer, price
    price_layer = Dense(output_dim, kernel_initializer=VarianceScaling())
    
    model.add(price_layer)

    return model

def NN_generator_tanh(n_layers : int, n_nodes : int, input_dim : int, output_dim : int) -> Sequential:
    model = Sequential()

    # Layer 1, input
    layer1 = Dense(n_nodes, activation = 'tanh',
        kernel_initializer=VarianceScaling(), 
        input_shape = (input_dim,))
    model.add(layer1)
    
    # Hidden layers
    if n_layers > 1:
        for i in range(n_layers - 1):
            layer = Dense(n_nodes, activation = 'tanh',
                kernel_initializer=VarianceScaling())
            model.add(layer)
    
    # Output layer, price
    price_layer = Dense(output_dim, kernel_initializer=VarianceScaling())
    
    model.add(price_layer)

    return model