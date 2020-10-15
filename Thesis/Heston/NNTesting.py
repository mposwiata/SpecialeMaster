import numpy as np
import itertools
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from Thesis.misc import VanillaOptions as vo
from Thesis.Heston.DataGeneration import calcImpVol

model_single_1 = load_model("Models/HestonSinglePrice/Heston_imp_single_1.h5")

norm_feature_single_1 = joblib.load("Models/HestonSinglePrice/norm_features.pkl")
norm_labels_single_1 = joblib.load("Models/HestonSinglePrice/norm_labels.pkl")

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
maturity = np.linspace(start = 0.01, stop = 2, num = 5)

# strike
strike = np.linspace(start = 75, stop = 125, num = 5)

option_input = np.array(list(itertools.product(maturity, strike))) # different option combinations
someOptionList = np.array([])
for option in option_input:
    someOptionList = np.append(someOptionList, vo.EUCall(option[0], option[1]))

# Dataframe for NN with multiple outputs
test1Data = np.array((forward, vol, kappa, theta, epsilon, rho, rate))
test1Data = np.reshape(test1Data, (1, 7))

# Model test for multiple outputs
testLength = np.shape(option_input)[0]
predictions_single = np.empty(testLength)
for i in range(testLength):
    testData = np.concatenate((test1Data, option_input[i]), axis=None)
    testData = np.reshape(testData, (1, -1))
    predictions_single[i] = norm_labels_single_1.inverse_transform(model_single_1.predict(norm_feature_single_1.transform(testData)))


# Generating benchmark data
benchmark = calcImpVol(test1Data[0], someOptionList)

fig, axs = plt.subplots(5, 2)

#axs[0].plot(option_input[0:5, 1], predictions1[0, 0:5], color = "blue", label = "model1")
axs[0, 0].plot(option_input[0:5, 1], predictions_single[0:5], color = "red", label = "model2")
axs[0, 0].plot(option_input[0:5, 1], benchmark[0:5], color = "black", label = "benchmark")
#axs[0].legend(loc="upper left")

#axs[1].plot(option_input[0:5, 1], predictions1[0, 5:10], color = "blue", label = "model1")
axs[1, 0].plot(option_input[0:5, 1], predictions_single[5:10], color = "red", label = "model2")
axs[1, 0].plot(option_input[0:5, 1], benchmark[5:10], color = "black", label = "benchmark")
#axs[1].legend(loc="upper left")

#axs[2].plot(option_input[0:5, 1], predictions1[0, 10:15], color = "blue", label = "model1")
axs[2, 0].plot(option_input[0:5, 1], predictions_single[10:15], color = "red", label = "model2")
axs[2, 0].plot(option_input[0:5, 1], benchmark[10:15], color = "black", label = "benchmark")
#axs[2].legend(loc="upper left")

#axs[3].plot(option_input[0:5, 1], predictions1[0, 15:20], color = "blue", label = "model1")
axs[3, 0].plot(option_input[0:5, 1], predictions_single[15:20], color = "red", label = "model2")
axs[3, 0].plot(option_input[0:5, 1], benchmark[15:20], color = "black", label = "benchmark")
#axs[3].legend(loc="upper left")

#axs[4].plot(option_input[0:5, 1], predictions1[0, 20:25], color = "blue", label = "model1")
axs[4, 0].plot(option_input[0:5, 1], predictions_single[20:25], color = "red", label = "model2")
axs[4, 0].plot(option_input[0:5, 1], benchmark[20:25], color = "black", label = "benchmark")
#axs[4].legend(loc="upper left")

#axs2[0].plot(option_input[0:5, 1], predictions_grid_1[0, 0:5] - benchmark[0:5], color = "blue", label = "model1")
axs[0, 1].plot(option_input[0:5, 1], predictions_single[0:5] - benchmark[0:5], color = "red", label = "model2")
#axs[0].legend(loc="upper left")

#axs2[1].plot(option_input[0:5, 1], predictions_grid_1[0, 5:10] - benchmark[5:10], color = "blue", label = "model1")
axs[1, 1].plot(option_input[0:5, 1], predictions_single[5:10] - benchmark[5:10], color = "red", label = "model2")
#axs[1].legend(loc="upper left")

#axs2[2].plot(option_input[0:5, 1], predictions_grid_1[0, 10:15] - benchmark[10:15], color = "blue", label = "model1")
axs[2, 1].plot(option_input[0:5, 1], predictions_single[10:15] - benchmark[10:15], color = "red", label = "model2")
#axs[2].legend(loc="upper left")

#axs2[3].plot(option_input[0:5, 1], predictions_grid_1[0, 15:20] - benchmark[15:20], color = "blue", label = "model1")
axs[3, 1].plot(option_input[0:5, 1], predictions_single[15:20] - benchmark[15:20], color = "red", label = "model2")
#axs[3].legend(loc="upper left")

#axs2[4].plot(option_input[0:5, 1], predictions_grid_1[0, 20:25] - benchmark[20:25], color = "blue", label = "model1")
axs[4, 1].plot(option_input[0:5, 1], predictions_single[20:25] - benchmark[20:25], color = "red", label = "model2")
#axs[4].legend(loc="upper left")

plt.savefig("HestonModelTest.jpeg")
plt.show()
