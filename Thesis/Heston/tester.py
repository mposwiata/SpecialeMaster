import numpy as np
import itertools
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
import glob
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam
import os
import itertools
import pickle
import sys
sys.path.append(os.getcwd()) # added for calc server support

from Thesis.Heston import AndersenLake as al, HestonModel as hm, DataGeneration as dg, ModelGenerator as mg
from Thesis.misc import VanillaOptions as vo

model_string = "Models5/standardize_non_early/standardize_non_early_1_50.h5"

train_index, test_index = mg.load_index(200000)
model_input = np.loadtxt("Data/benchmark_input.csv", delimiter = ",")
imp_vol = np.loadtxt("Data/benchmark_imp.csv", delimiter=",")
price = np.loadtxt("Data/benchmark_price.csv", delimiter=",")

X_test = model_input[test_index, :]
Y_test = imp_vol[test_index, :]
Y_test_price = price[test_index, :]

if (model_string.find("price") != -1):
    y_test_loop = Y_test_price
else:
    y_test_loop = Y_test

x_test_loop = X_test

if (model_string.find("single") != -1):
    x_test_loop, y_test_loop = mg.transform_single(x_test_loop, y_test_loop)
elif (model_string.find("mat") != -1):
    x_test_loop, y_test_loop = mg.transform_mat(x_test_loop, y_test_loop)

if ((model_string.find("benchmark_include") != -1) or (model_string.find("price_include") != -1)):
    index = np.all(y_test_loop != -1, axis = 1)
else:
    index = np.all(y_test_loop > 0, axis = 1)

x_test_loop = x_test_loop[index, :]
y_test_loop = y_test_loop[index, :]

model = load_model(model_string)
model_folder = model_string[:model_string.rfind("/") + 1]
if os.path.exists(model_folder+"/norm_feature.pkl"):
    norm_feature = joblib.load(model_folder+"norm_feature.pkl")
    x_test_loop = norm_feature.transform(x_test_loop)
if os.path.exists(model_folder+"/norm_labels.pkl"):
    norm_labels = joblib.load(model_folder+"norm_labels.pkl")
    y_test_loop = norm_labels.transform(y_test_loop)

name = model_string[model_string.rfind("/")+1:]
score = model.evaluate(x_test_loop, y_test_loop, verbose=0)
print(score)