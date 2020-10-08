import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from keras import backend as k
from Thesis import DataGenerator as dg, NeuralNetworkGenerator as nng
from Thesis.Models import black_scholes as bs

strike = 100
sigma = 0.2 
mat = 1

X, Y = dg.BSDataGenerator(strike, sigma, mat, 300000)

test_X, test_Y = dg.BSDataGenerator(strike, sigma, mat, 50000)

norm_features = StandardScaler() #MinMaxScaler(feature_range = (-1, 1))
norm_labels = StandardScaler()

norm_x = norm_features.fit_transform(X)
norm_y = norm_labels.fit_transform(Y)

norm_test_x = norm_features.transform(test_X)
norm_test_y = norm_labels.transform(test_Y)

eval_set = [(norm_test_x, norm_test_y)]

xg_reg = xgb.XGBRegressor(learning_rate = 0.1, booster = 'gbtree')

xg_reg.fit(norm_x, norm_y, eval_metric = "rmse", eval_set = eval_set, verbose = True)

#rmse = np.sqrt(mean_squared_error(y_test, preds))
#print("RMSE: %f" % (rmse))

# Generating benchmark data
spotPlot = np.linspace(start = 1, stop = 200, num = 1000)

pricePlot = bs.vBlackScholesFormula(spotPlot, strike, mat, sigma, 0)
price = pricePlot.reshape(-1, 1)
spotNorm = spotPlot.reshape(-1, 1)

norm_spot = norm_features.transform(spotNorm)

preds = norm_labels.inverse_transform(xg_reg.predict(norm_spot))

"""
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
"""
plt.scatter(spotPlot, preds)
plt.plot(spotPlot, pricePlot, 'g-')
plt.savefig("blackScholesXG.jpeg")