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

model1 = load_model('sabr1.h5')
predictions1 = np.reshape(model1.predict(data), (10,))

model2 = load_model('sabr2.h5')
predictions2 = np.reshape(model2.predict(data), (10,))

model3 = load_model('sabr3.h5')
predictions3 = np.reshape(model3.predict(data), (10,))

model4 = load_model('sabr4.h5')
predictions4 = np.reshape(model4.predict(data), (10,))

model5 = load_model('sabr5.h5')
predictions5 = np.reshape(model5.predict(data), (10,))

model6 = load_model('sabr6.h5')
predictions6 = np.reshape(model6.predict(data), (10,))

model7 = load_model('sabr7.h5')
predictions7 = np.reshape(model7.predict(data), (10,))

model8 = load_model('sabr8.h5')
predictions8 = np.reshape(model8.predict(data), (10,))


plt.plot(strike, predictions1, color = "blue", label = "1")
plt.plot(strike, predictions2, color = "red", label = "2")
plt.plot(strike, predictions3, color = "black", label = "3")
plt.plot(strike, predictions4, color = "green", label = "4")
plt.plot(strike, predictions5, color = "orange", label = "5")
plt.plot(strike, predictions6, color = "pink", label = "6")
plt.plot(strike, predictions7, color = "yellow", label = "7")
plt.plot(strike, predictions8, color = "brown", label = "8")
plt.plot(strike, approx, color = "grey", label = "Approx")
plt.legend(loc="upper left")
plt.savefig("predictions_sabr.jpeg")
