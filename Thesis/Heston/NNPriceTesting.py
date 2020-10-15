import numpy as np
import itertools
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Thesis.misc import VanillaOptions as vo
from Thesis.Heston.PriceGeneration import calcPrice

# importing models
model_grid_1 = load_model("Models/HestonGridPrice/Heston_price_grid_1.h5")
model_grid_2 = load_model("Models/HestonGridPrice/Heston_price_grid_2.h5")

# importing scalars
norm_feature_grid_1 = joblib.load("Models/HestonGridPrice/norm_features_1.pkl")
norm_labels_grid_1 = joblib.load("Models/HestonGridPrice/norm_labels_1.pkl")

norm_feature_grid_2 = joblib.load("Models/HestonGridPrice/norm_features_2.pkl")
norm_labels_grid_2 = joblib.load("Models/HestonGridPrice/norm_labels_1.pkl")

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
predictions_grid_1 = norm_labels_grid_1.inverse_transform(model_grid_1.predict(norm_feature_grid_1.transform(test1Data)))
predictions_grid_2 = norm_labels_grid_2.inverse_transform(model_grid_2.predict(norm_feature_grid_2.transform(test1Data)))

# Generating benchmark data
benchmark = calcPrice(test1Data[0], someOptionList)

# Generating price plot
fig, axs = plt.subplots(5,2)

#axs[0].plot(option_input[0:5, 1], predictions_grid_1[0, 0:5], color = "blue", label = "model1")
axs[0, 0].plot(option_input[0:5, 1], predictions_grid_2[0, 0:5], color = "red", label = "model2")
axs[0, 0].plot(option_input[0:5, 1], benchmark[0:5], color = "black", label = "benchmark")
#axs[0].legend(loc="upper left")

#axs[1].plot(option_input[0:5, 1], predictions_grid_1[0, 5:10], color = "blue", label = "model1")
axs[1, 0].plot(option_input[0:5, 1], predictions_grid_2[0, 5:10], color = "red", label = "model2")
axs[1, 0].plot(option_input[0:5, 1], benchmark[5:10], color = "black", label = "benchmark")
#axs[1].legend(loc="upper left")

#axs[2].plot(option_input[0:5, 1], predictions_grid_1[0, 10:15], color = "blue", label = "model1")
axs[2, 0].plot(option_input[0:5, 1], predictions_grid_2[0, 10:15], color = "red", label = "model2")
axs[2, 0].plot(option_input[0:5, 1], benchmark[10:15], color = "black", label = "benchmark")
#axs[2].legend(loc="upper left")

#axs[3].plot(option_input[0:5, 1], predictions_grid_1[0, 15:20], color = "blue", label = "model1")
axs[3, 0].plot(option_input[0:5, 1], predictions_grid_2[0, 15:20], color = "red", label = "model2")
axs[3, 0].plot(option_input[0:5, 1], benchmark[15:20], color = "black", label = "benchmark")
#axs[3].legend(loc="upper left")

#axs[4].plot(option_input[0:5, 1], predictions_grid_1[0, 20:25], color = "blue", label = "model1")
axs[4, 0].plot(option_input[0:5, 1], predictions_grid_2[0, 20:25], color = "red", label = "model2")
axs[4, 0].plot(option_input[0:5, 1], benchmark[20:25], color = "black", label = "benchmark")
#axs[4].legend(loc="upper left")

#axs2[0].plot(option_input[0:5, 1], predictions_grid_1[0, 0:5] - benchmark[0:5], color = "blue", label = "model1")
axs[0, 1].plot(option_input[0:5, 1], predictions_grid_2[0, 0:5] - benchmark[0:5], color = "red", label = "model2")
#axs[0].legend(loc="upper left")

#axs2[1].plot(option_input[0:5, 1], predictions_grid_1[0, 5:10] - benchmark[5:10], color = "blue", label = "model1")
axs[1, 1].plot(option_input[0:5, 1], predictions_grid_2[0, 5:10] - benchmark[5:10], color = "red", label = "model2")
#axs[1].legend(loc="upper left")

#axs2[2].plot(option_input[0:5, 1], predictions_grid_1[0, 10:15] - benchmark[10:15], color = "blue", label = "model1")
axs[2, 1].plot(option_input[0:5, 1], predictions_grid_2[0, 10:15] - benchmark[10:15], color = "red", label = "model2")
#axs[2].legend(loc="upper left")

#axs2[3].plot(option_input[0:5, 1], predictions_grid_1[0, 15:20] - benchmark[15:20], color = "blue", label = "model1")
axs[3, 1].plot(option_input[0:5, 1], predictions_grid_2[0, 15:20] - benchmark[15:20], color = "red", label = "model2")
#axs[3].legend(loc="upper left")

#axs2[4].plot(option_input[0:5, 1], predictions_grid_1[0, 20:25] - benchmark[20:25], color = "blue", label = "model1")
axs[4, 1].plot(option_input[0:5, 1], predictions_grid_2[0, 20:25] - benchmark[20:25], color = "red", label = "model2")
#axs[4].legend(loc="upper left")

plt.savefig("HestonTest3.jpeg")
plt.show()