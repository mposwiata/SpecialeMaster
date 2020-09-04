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
from Thesis.Models import black_scholes as bs

model = load_model('first_model.h5')

spot = np.arange(10, 200, 2)
data_len = np.shape(spot)[0]
spot = np.reshape(spot, (data_len, 1))

K = 100

spot = spot / K
K = np.full([data_len, 1], 1)

r = np.full([data_len, 1], 0.05)
sigma = np.full([data_len, 1], 0.2)
mat = np.full([data_len, 1], 1)

data = np.concatenate((spot, K, mat, sigma), 1)

bs_values = np.empty(data_len)
i = 0
for spot_value in spot:
    bs_values[i] = bs.BlackScholesFormula(spot[i], 1, 1, 0.2, 0.05)
    i += 1

predictions = model.predict(data)
plt.plot(spot, predictions)
plt.plot(spot, bs_values)
plt.savefig("predictions.jpeg")

