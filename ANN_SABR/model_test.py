import numpy as np
import tensorflow
import matplotlib.pyplot as plt
from keras.models import load_model
from Thesis.Functions import hagan
from Thesis.Functions import SABR_ANN as sa

mat = 0.5
sigma_0 = 0.3
nu = 0.4
rho = 0
strike_limit = sa.strike_par(mat, sigma_0, rho, nu)
strike = np.linspace(start = strike_limit[0], stop = strike_limit[1], num = 10)

data = np.array((mat, sigma_0, rho, nu))
data = np.append(data, strike)
data = np.reshape(data, (1, 14))

approx = np.empty(10)
i = 0
for someStrike in strike:
    approx[i] = hagan.hagan_sigma_b(mat, sigma_0, rho, nu, someStrike)
    i += 1

model1 = load_model('testSABRModel.h5')
predictions1 = np.reshape(model1.predict(data), (10,))

plt.plot(strike, predictions1, color = "blue", label = "1")
plt.plot(strike, approx, color = "grey", label = "Approx")
plt.legend(loc="upper left")
plt.savefig("predictions_sabr.jpeg")
