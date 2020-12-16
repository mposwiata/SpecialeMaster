import numpy as np
import time
import matplotlib.pyplot as plt

from Thesis.Heston import AndersenLake as al, MonteCarlo as mc, HestonModel as hm
from Thesis.misc import VanillaOptions as vo

model_input = np.loadtxt("Data/benchmark_input.csv", delimiter = ",")
#imp_vol = np.loadtxt("Data/benchmark_input.csv", delimiter = ",")
some_model = hm.HestonClass(*model_input[0])
some_option = vo.EUCall(0.1, 87.5)

price = al.Andersen_Lake(some_model, some_option)
print(price)
imp_vol = some_model.impVol(price, some_option)
print(imp_vol)