import numpy as np
import itertools
import matplotlib.pyplot as plt
import joblib
import time
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Thesis.misc import VanillaOptions as vo
from Thesis.Heston.PriceGeneration import calcPrice
from Thesis.Heston import AndersenLake as al, HestonModel as hm

# importing models
model_grid_1 = load_model("Models/HestonGridPrice/Heston_price_grid_1.h5")
model_grid_2 = load_model("Models/HestonGridPrice/Heston_price_grid_2.h5")
model_single_1 = load_model("Models/HestonSinglePrice/Heston_price_single_1.h5")

# importing scalars
norm_feature_grid_1 = joblib.load("Models/HestonGridPrice/norm_features_1.pkl")
norm_labels_grid_1 = joblib.load("Models/HestonGridPrice/norm_labels_1.pkl")

norm_feature_grid_2 = joblib.load("Models/HestonGridPrice/norm_features_2.pkl")
norm_labels_grid_2 = joblib.load("Models/HestonGridPrice/norm_labels_1.pkl")

norm_feature_single_1 = joblib.load("Models/HestonSinglePrice/norm_features_price_1.pkl")
norm_labels_single_1 = joblib.load("Models/HestonSinglePrice/norm_labels_price_1.pkl")

# Model inputs for test
# Forward
forward = 100

# vol
vol = 0.1

# kappa
kappa = 0.5

# theta
theta = 0.1

# epsilon
epsilon = 1

# rho
rho = -0.5

# rate
rate = 0.05

# Which options to test
# Maturity
no_mats = 10
maturity = np.linspace(start = 0.01, stop = 2, num = no_mats)

# strike
no_strikes = 10
strike = np.linspace(start = 75, stop = 125, num = no_strikes)

option_input = np.array(list(itertools.product(maturity, strike))) # different option combinations
someOptionList = np.array([])
for option in option_input:
    someOptionList = np.append(someOptionList, vo.EUCall(option[0], option[1]))

# Dataframe for NN with multiple outputs
test1Data = np.array((forward, vol, kappa, theta, epsilon, rho, rate))
test1Data = np.reshape(test1Data, (1, 7))

# Model test for multiple outputs
predictions_grid_1 = norm_labels_grid_1.inverse_transform(model_grid_1.predict(norm_feature_grid_1.transform(test1Data)))
predictions_grid_2 = norm_labels_grid_2.inverse_transform(model_grid_2.predict(norm_feature_grid_2.transform(test1Data)))

# Model test for single outputs
testLength = np.shape(option_input)[0]
predictions_single = np.empty(testLength)
start_predict = time.time()
for i in range(testLength):
    testData = np.concatenate((test1Data, option_input[i]), axis=None)
    testData = np.reshape(testData, (1, -1))
    predictions_single[i] = norm_labels_single_1.inverse_transform(model_single_1.predict(norm_feature_single_1.transform(testData)))
stop_predict = time.time()
#print("Prediction time: ", stop_predict - start_predict)

# Generating benchmark data
start_bench = time.time()
benchmark = calcPrice(test1Data[0], someOptionList)
stop_bench = time.time()
#print("Benchmark time: ", stop_bench - start_bench)
fig, axs = plt.subplots(no_mats, 2)
for i in range(no_mats):
    axs[i, 0].plot(option_input[0 : no_strikes, 1], predictions_single[no_strikes * i : no_strikes * (i + 1)], color = "red", alpha = 0.5, label = "model1")
    axs[i, 0].plot(option_input[0 : no_strikes, 1], benchmark[no_strikes * i : no_strikes * (i + 1)], color = "black", alpha = 0.7, label = "benchmark")
    axs[i, 1].plot(option_input[0 : no_strikes, 1], predictions_single[no_strikes * i : no_strikes * (i + 1)] - benchmark[no_strikes * i : no_strikes * (i + 1)], color = "black")

plt.savefig("HestonGridSingleTest.jpeg")

# Calculation times
testDataSet = np.array([], dtype=np.int64).reshape(0, 9)
for i in range(32):
    testData = np.concatenate((test1Data, option_input[i]), axis=None)
    testData = np.reshape(testData, (1, -1))
    testDataSet = np.concatenate((testDataSet, testData), axis = 0)



transformed_data = norm_feature_single_1.transform(testDataSet)
predict_start_test = time.time()
norm_labels_single_1.inverse_transform(model_single_1.predict(transformed_data, batch_size=32))
predict_stop_test = time.time()
print("Prediction time: ", predict_stop_test - predict_start_test)

someModel = hm.HestonClass(test1Data[0, 0], test1Data[0, 1], test1Data[0, 2], test1Data[0, 3], test1Data[0, 4], test1Data[0, 5], test1Data[0, 6])
bench_start_test = time.time()
calcPrice(test1Data[0], someOptionList[0:32])
bench_stop_test = time.time()
print("Benchmark time: ", bench_stop_test - bench_start_test)