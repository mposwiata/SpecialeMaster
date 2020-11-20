import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam
from keras import backend as k
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from multiprocessing import Pool, cpu_count
from scipy.special import ndtr
from scipy.stats import norm
import sys
import os
sys.path.append(os.getcwd()) # added for calc server support

from Thesis.Heston import MonteCarlo as mc, AndersenLake as al, HestonModel as hm, NNModelGenerator as mg
from Thesis.misc import VanillaOptions as vo
from Thesis import NeuralNetworkGenerator as nng
from Thesis.BlackScholes import BlackScholes as bs

### Generating input data
### Generating input data
spot = np.linspace(start = 50, stop = 150, num = 1000)
vol1 = 0.04
vol2 = 0.01
kappa1 = 2
kappa2 = 0.1
theta1 = 0.04
theta2 = 0.01
epsilon1 = 0.5
epsilon2 = 2
epsilon = np.linspace(start = 0.5, stop = 2, num = 10)
rho1 = -0.7
rho2 = 0.8
rate = 0.05
input_array = np.array(list(itertools.product(spot, epsilon)))

tau = 1.005 #set to match option data
strike = 100

some_option = vo.EUCall(tau, strike)

spot = np.reshape(spot, (-1, 1))

spot_plot = np.linspace(start = 75, stop = 125, num = 200)
spot_plot = np.reshape(spot_plot, (-1, 1))
eps1_plot = np.reshape(np.repeat(epsilon1, len(spot_plot)), (-1, 1))
eps2_plot = np.reshape(np.repeat(epsilon2, len(spot_plot)), (-1, 1))
input_multi_easy = np.concatenate([spot_plot, eps1_plot], axis = 1)
input_multi_hard = np.concatenate([spot_plot, eps2_plot], axis = 1)
h = 0.1
input_good_easy = np.concatenate([
    spot_plot, 
    np.reshape(np.repeat(vol1, len(spot_plot)), (-1, 1)),
    np.reshape(np.repeat(kappa1, len(spot_plot)), (-1, 1)),
    np.reshape(np.repeat(theta1, len(spot_plot)), (-1, 1)),
    np.reshape(np.repeat(epsilon1, len(spot_plot)), (-1, 1)),
    np.reshape(np.repeat(rho1, len(spot_plot)), (-1, 1)),
    np.reshape(np.repeat(rate, len(spot_plot)), (-1, 1))
], axis = 1)
input_good_easy_low = np.copy(input_good_easy)
input_good_easy_low[:, 0] = input_good_easy_low[:, 0] - h
input_good_hard = np.concatenate([
    spot_plot, 
    np.reshape(np.repeat(vol2, len(spot_plot)), (-1, 1)),
    np.reshape(np.repeat(kappa2, len(spot_plot)), (-1, 1)),
    np.reshape(np.repeat(theta2, len(spot_plot)), (-1, 1)),
    np.reshape(np.repeat(epsilon2, len(spot_plot)), (-1, 1)),
    np.reshape(np.repeat(rho2, len(spot_plot)), (-1, 1)),
    np.reshape(np.repeat(rate, len(spot_plot)), (-1, 1))
], axis = 1) 
input_good_easy_high = np.copy(input_good_easy)
input_good_easy_high[:, 0] = input_good_easy_high[:, 0] + h

### Testing the thesis model
#good_model= load_model()
norm_folder = "Models4/norms/"
norm_feature_good = joblib.load(norm_folder+"norm_feature.pkl")
model = load_model("Models4/activation_functions/mix_5_1000.h5")

a = norm_feature_good.data_min_[0]
b = norm_feature_good.data_max_[0]

inp_tensor_easy = tf.convert_to_tensor(norm_feature_good.transform(input_good_easy))
inp_tensor_hard = tf.convert_to_tensor(norm_feature_good.transform(input_good_hard))

### Andersen Lake model
with tf.GradientTape(persistent = True) as tape:
    tape.watch(inp_tensor_easy)
    with tf.GradientTape(persistent = True) as tape2:
        tape2.watch(inp_tensor_easy)
        predict_easy = model(inp_tensor_easy)[:,12]
    grads_easy = tape2.gradient(predict_easy, inp_tensor_easy)[:,0]

grads2_easy = tape.gradient(grads_easy, inp_tensor_easy).numpy()
grads2_easy = grads2_easy[:,0]
grads_easy = grads_easy.numpy()

with tf.GradientTape(persistent = True) as tape:
    tape.watch(inp_tensor_hard)
    with tf.GradientTape(persistent = True) as tape2:
        tape2.watch(inp_tensor_hard)
        predict_hard = model(inp_tensor_hard)[:,12]
    grads_hard = tape2.gradient(predict_hard, inp_tensor_hard)[:,0]

grads2_hard = tape.gradient(grads_hard, inp_tensor_hard).numpy()
grads2_hard = grads2_hard[:,0]
grads_hard = grads_hard.numpy()

delta_easy = np.zeros(200)
price_easy = np.zeros(200)
gamma_easy = np.zeros(200)
delta_hard = np.zeros(200)
price_hard = np.zeros(200)
gamma_hard = np.zeros(200)

delta = np.zeros(200)
delta2 = np.zeros(200)
gamma = np.zeros(200)
gamma2 = np.zeros(200)

for i in range(200):
    model_bs_easy = bs.BlackScholesForward(spot_plot[i], predict_easy[i], rate)
    model_bs_hard = bs.BlackScholesForward(spot_plot[i], predict_hard[i], rate)
    price_easy[i] = model_bs_easy.BSFormula(some_option)
    """
    delta[i] = (price_easy[i] - price_easy[i-1]) / (spot_plot[i] - spot_plot[i-1])
    if i > 2:
        gamma[i] = (delta[i] - delta[i-1]) / (spot_plot[i] - spot_plot[i-1])
    delta_easy[i] = model_bs_easy.delta_grads(some_option, a, b, grads_easy[i])
    gamma_easy[i] = model_bs_easy.gamma_grads(some_option, a, b, grads_easy[i], grads2_easy[i])
    """
    
    #price_hard[i] = model_bs_hard.BSFormula(some_option)
    #delta_hard[i] = model_bs_hard.delta_grads(some_option, a, b, grads_hard[i])
    #gamma_hard[i] = model_bs_hard.gamma_grads(some_option, a, b, grads_hard[i], grads2_hard[i])

for i in range(1, 199):
    delta[i] = (price_easy[i] - price_easy[i-1]) / (spot_plot[i] - spot_plot[i-1])
    delta2[i] = (price_easy[i+1] - price_easy[i-1]) / (spot_plot[i+1] - spot_plot[i-1])

for i in range(1, 199):
    gamma2[i] = (delta[i+1] - delta[i-1]) / (spot_plot[i+1] - spot_plot[i-1])
    gamma[i] = (delta[i] - delta[i-1]) / (spot_plot[i] - spot_plot[i-1])

