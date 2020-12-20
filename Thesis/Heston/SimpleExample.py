import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
import joblib
from keras.models import load_model
import glob
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam
import os
import itertools
import pickle

from Thesis.Heston import AndersenLake as al, HestonModel as hm, DataGeneration as dg, ModelGenerator as mg, MC_al, ModelTesting as mt
from Thesis.misc import VanillaOptions as vo
from Thesis.BlackScholes import BlackScholes as bs

some_easy_case = mt.easy_case()
some_heston_model = hm.HestonClass(*some_easy_case[0])
some_option = vo.EUCall(2, 75)

al.Andersen_Lake(some_heston_model, some_option)
