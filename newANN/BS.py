import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
sigma = 0.4
mat = 1

X, Y = dg.BSDataGenerator(strike, sigma, mat, 30000)

norm_features = MinMaxScaler(feature_range = (-1, 1))
norm_labels = StandardScaler()

norm_x = norm_features.fit_transform(X)
norm_y = norm_labels.fit_transform(Y)

model = nng.NNGenerator(4, 5, 1)

adam = Adam(lr = 0.1)

model.compile(
    loss = 'mean_squared_error', #mean squared error
    optimizer = adam
    )

callbacks_list = [
    LearningRateScheduler(lr_schedule, verbose = 0),
    EarlyStopping(monitor='val_loss', patience=15)
]

model.fit(norm_x, norm_y, epochs=100, batch_size=1024, verbose = 0, callbacks = callbacks_list, validation_split = 0.1)

model.save("testBSModel.h5")

spotPlot = np.linspace(start = 1, stop = 200, num = 1000)

pricePlot = bs.vBlackScholesFormula(spotPlot, strike, mat, sigma, 0)
price = pricePlot.reshape(-1, 1)
spotNorm = spotPlot.reshape(-1, 1)

norm_spot = norm_features.transform(spotNorm)

predictions = norm_labels.inverse_transform(model.predict(norm_spot))

plt.scatter(X, Y, c = 'grey', alpha = 0.5, s = 0.5)
plt.plot(spotPlot, pricePlot, 'r-')
plt.plot(spotPlot, predictions, 'b--')
plt.xlim(0, 180)
plt.ylim(0, 80)

plt.savefig("blackScholesPlot.jpeg")