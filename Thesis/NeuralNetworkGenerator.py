import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, GaussianNoise, core
from keras.initializers import VarianceScaling
from keras import regularizers

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

def NN_generator_mix(n_layers : int, n_nodes : int, input_dim : int, output_dim : int) -> Sequential:
    model = Sequential()

    # Layer 1, input
    layer1 = Dense(n_nodes, activation = 'tanh',
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

def NN_generator_mix_noise(n_layers : int, n_nodes : int, input_dim : int, output_dim : int) -> Sequential:
    model = Sequential()

    model.add(GaussianNoise(0.1, input_shape = (input_dim,)))

    # Layer 1, input
    layer1 = Dense(n_nodes, activation = 'tanh',
        kernel_initializer=VarianceScaling())
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

def NN_generator_regul(n_layers : int, n_nodes : int, input_dim : int, output_dim : int) -> Sequential:
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
                kernel_initializer=VarianceScaling(),kernel_regularizer=regularizers.l2(0.01))
            model.add(layer)
    
    # Output layer, price
    price_layer = Dense(output_dim, kernel_initializer=VarianceScaling())
    
    model.add(price_layer)

    return model

def NN_generator_dropout(n_layers : int, n_nodes : int, input_dim : int, output_dim : int) -> Sequential:
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
            dropout = core.Dropout(0.25)
            model.add(layer)
            model.add(dropout)
    
    # Output layer, price
    price_layer = Dense(output_dim, kernel_initializer=VarianceScaling())
    
    model.add(price_layer)

    return model
