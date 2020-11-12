import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras import backend as k
from multiprocess import Pool, cpu_count, Process
import joblib
import sys
import os
sys.path.append(os.getcwd()) # added for calc server support

from Thesis import NeuralNetworkGenerator as nng
from sklearn.model_selection import train_test_split
from Thesis.Heston import NNModelGenerator as mg
from Thesis.Heston import DataGeneration as dg

### Sobol, 200.000
sobol1 = np.loadtxt("Data/hestonSobolGridInput2_200000.csv", delimiter = ",")
sobol1_imp = np.loadtxt("Data/hestonSobolGridImpVol2_200000.csv", delimiter = ",")

sobol1_filter = np.all(sobol1_imp != 0, axis = 1)
sobol1_input = sobol1[sobol1_filter, :]
sobol1_output = sobol1_imp[sobol1_filter, :]

### sobol, new
sobol2 = np.loadtxt("Data/hestonSobolGridInput2_compare_200000.csv", delimiter = ",")
sobol2_imp = np.loadtxt("Data/hestonSobolGridImpVol2_compare_200000.csv", delimiter = ",")

sobol2_filter = np.all(sobol2_imp != 0, axis = 1)
sobol2_input = sobol1[sobol2_filter, :]
sobol2_output = sobol1_imp[sobol2_filter, :]