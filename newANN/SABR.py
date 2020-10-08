import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras import backend as k
from Thesis import DataGenerator as dg, NeuralNetworkGenerator as nng
from Thesis.Models import black_scholes as bs

def lr_schedule(n, alpha):
    a, b = 1e-4, 1e-2
    n1, n2, n3 = 0, 24, 74

    if n <= n2:
        return (a - b)/(n1 - n2) * n - (a*n2 - b*n1) / (n1 - n2)
    elif n2 < n < n3:
        return -(a - b)/(n2 - n3) * n + (a*n2 - b*n3) / (n2 - n3)
    else:
        return a

strike = 100
sigma = 0.2 
mat = 1

train_input = np.loadtxt("./Data/train_input_sabr.csv", delimiter=",")
train_output = np.loadtxt("./Data/train_output_sabr_approx.csv", delimiter=",")
test_input = np.loadtxt("./Data/test_input_sabr.csv", delimiter=",")
test_output = np.loadtxt("./Data/test_output_sabr_approx.csv", delimiter=",")

norm_features = StandardScaler() #MinMaxScaler(feature_range = (-1, 1))
norm_labels = StandardScaler()

norm_x = norm_features.fit_transform(train_input)
norm_y = norm_labels.fit_transform(train_output)

norm_x_test = norm_features.transform(test_input)
norm_y_test = norm_labels.transform(test_output)

model = nng.NNGenerator(4, 5, 14, 10)

adam = Adam(lr = 0.1)

model.compile(
    loss = 'mean_squared_error', #mean squared error
    optimizer = adam
    )

callbacks_list = [
    LearningRateScheduler(lr_schedule, verbose = 0),
    EarlyStopping(monitor='val_loss', patience=25)
]

model.fit(norm_x, norm_y, epochs=100, batch_size=32, verbose = 2, callbacks = callbacks_list, validation_data=(norm_x_test, norm_y_test))

model.save("testSABRModel.h5")

"""
spotPlot = np.linspace(start = 1, stop = 200, num = 1000)

pricePlot = bs.vBlackScholesFormula(spotPlot, strike, mat, sigma, 0)
price = pricePlot.reshape(-1, 1)
spotNorm = spotPlot.reshape(-1, 1)

# Getting delta's from the formula
delta = bs.vBlackScholesDelta(spotPlot, strike, mat, sigma, 0)

norm_spot = norm_features.transform(spotNorm)

predictions = norm_labels.inverse_transform(model.predict(norm_spot))

# Getting the derivatives from the NN
inp_tensor = tf.convert_to_tensor(norm_spot)

with tf.GradientTape() as tape:
    tape.watch(inp_tensor)
    predict = model(inp_tensor)

grads = tape.gradient(predict, inp_tensor)

grads2 = grads * np.sqrt(norm_labels.var_) / np.sqrt(norm_features.var_)

plt.plot(spotPlot, grads2, 'r--')
plt.plot(spotPlot, delta, 'g-')
plt.savefig("blackScholesDelta.jpeg")
"""