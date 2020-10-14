import numpy as np
import itertools
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from Thesis.misc import VanillaOptions as vo
from Thesis.Heston.DataGeneration import calcImpVol

input1 = np.loadtxt("Data/hestonGridInput.csv", delimiter=",")
output1 = np.loadtxt("Data/hestonGridOutput.csv", delimiter=",")

input2 = np.loadtxt("Data/hestonSingleInput.csv", delimiter=",")
output2 = np.loadtxt("Data/hestonSingleOutput.csv", delimiter=",")
output_data = np.reshape(output2, (-1, 1))

X1_train, X1_test, y1_train, y1_test = train_test_split(input1, output1, test_size=0.3, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(input2, output_data, test_size=0.3, random_state=42)

norm_features1 = StandardScaler() #MinMaxScaler(feature_range = (-1, 1))
norm_labels1 = StandardScaler()
norm_features2 = StandardScaler() #MinMaxScaler(feature_range = (-1, 1))
norm_labels2 = StandardScaler()

norm_features1.fit(X1_train)
norm_labels1.fit(y1_train)

norm_features2.fit(X2_train)
norm_labels2.fit(y2_train)

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
epsilon = 0.5

# rho
rho = 0.3

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
model1 = load_model('Heston_imp_1.h5')

predictions1 = norm_labels1.inverse_transform(model1.predict(norm_features1.transform(test1Data)))

model2 = load_model('testHestonSingleModel.h5')
testLength = np.shape(option_input)[0]
predictions2 = np.empty(testLength)
for i in range(testLength):
    testData = np.concatenate((test1Data, option_input[i]), axis=None)
    testData = np.reshape(testData, (1, -1))
    predictions2[i] = norm_labels2.inverse_transform(model2.predict(norm_features2.transform(testData)))

# Generating benchmark data
benchmark = calcImpVol(test1Data[0], someOptionList)

fig, axs = plt.subplots(5)

axs[0].plot(option_input[0:5, 1], predictions1[0, 0:5], color = "blue", label = "model1")
#axs[0].plot(option_input[0:5, 1], predictions2[0:5], color = "red", label = "model2")
axs[0].plot(option_input[0:5, 1], benchmark[0:5], color = "black", label = "benchmark")
#axs[0].legend(loc="upper left")

axs[1].plot(option_input[0:5, 1], predictions1[0, 5:10], color = "blue", label = "model1")
#axs[1].plot(option_input[0:5, 1], predictions2[5:10], color = "red", label = "model2")
axs[1].plot(option_input[0:5, 1], benchmark[5:10], color = "black", label = "benchmark")
#axs[1].legend(loc="upper left")

axs[2].plot(option_input[0:5, 1], predictions1[0, 10:15], color = "blue", label = "model1")
#axs[2].plot(option_input[0:5, 1], predictions2[10:15], color = "red", label = "model2")
axs[2].plot(option_input[0:5, 1], benchmark[10:15], color = "black", label = "benchmark")
#axs[2].legend(loc="upper left")

axs[3].plot(option_input[0:5, 1], predictions1[0, 15:20], color = "blue", label = "model1")
#axs[3].plot(option_input[0:5, 1], predictions2[15:20], color = "red", label = "model2")
axs[3].plot(option_input[0:5, 1], benchmark[15:20], color = "black", label = "benchmark")
#axs[3].legend(loc="upper left")

axs[4].plot(option_input[0:5, 1], predictions1[0, 20:25], color = "blue", label = "model1")
#axs[4].plot(option_input[0:5, 1], predictions2[20:25], color = "red", label = "model2")
axs[4].plot(option_input[0:5, 1], benchmark[20:25], color = "black", label = "benchmark")
#axs[4].legend(loc="upper left")

plt.savefig("HestonModelTest.jpeg")
plt.show()
