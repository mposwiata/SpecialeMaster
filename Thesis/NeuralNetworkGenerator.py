import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.initializers import VarianceScaling

def NNGenerator(nLayers : int, nNodes : int, inputDim : int, outputDim : int) -> Sequential:
    model = Sequential()

    # Layer 1, input
    layer1 = Dense(nNodes, activation = 'softplus',
        kernel_initializer=VarianceScaling(), 
        input_shape = (inputDim,))
    model.add(layer1)
    
    # Hidden layers
    if nLayers > 1:
        for i in range(nLayers - 1):
            layer = Dense(nNodes, activation = 'softplus',
                kernel_initializer=VarianceScaling())
            model.add(layer)
    
    # Output layer, price
    priceLayer = Dense(outputDim, kernel_initializer=VarianceScaling())
    
    model.add(priceLayer)

    return model