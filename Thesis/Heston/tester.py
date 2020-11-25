import numpy as np
import time
from Thesis.misc import VanillaOptions as vo 
from Thesis.Heston import NNModelGenerator as mg, HestonModel as hm, AndersenLake as al, DataGeneration as dg
from Thesis.BlackScholes import BlackScholes as bs

model_input = np.loadtxt("Data/MC/HestonMC_input.csv", delimiter=",")
price_1 = np.loadtxt("Data/MC/HestonMC_price_1.csv", delimiter=",")
price_10 = np.loadtxt("Data/MC/HestonMC_price_10.csv", delimiter=",")
price_100 = np.loadtxt("Data/MC/HestonMC_price_100.csv", delimiter=",")
price_1000 = np.loadtxt("Data/MC/HestonMC_price_1000.csv", delimiter=",")
price_10000 = np.loadtxt("Data/MC/HestonMC_price_10000.csv", delimiter=",")

option_array = dg.option_input_generator()
some_option_list = np.array([])
for option in option_array:
    some_option_list = np.append(some_option_list, vo.EUCall(option[0], option[1]))

output_imp_vol_1 = np.empty((200000, 25), dtype=np.float64)
output_imp_vol_10 = np.empty((200000, 25), dtype=np.float64)
output_imp_vol_100 = np.empty((200000, 25), dtype=np.float64)
output_imp_vol_1000 = np.empty((200000, 25), dtype=np.float64)
output_imp_vol_10000 = np.empty((200000, 25), dtype=np.float64)

def imp_vol(price : float, some_option : vo.VanillaOption) -> float:
    return some_model.impVol(price, some_option) if price != 0 else 0

j = 0
for some_row in model_input:
    some_model = hm.HestonClass(some_row[0], some_row[1], some_row[2], some_row[3], some_row[4], some_row[5], some_row[6])
    for i in range(len(some_option_list)):
        output_imp_vol_1[j, i] = imp_vol(price_1[j, i], some_option_list[i])
        output_imp_vol_10[j, i] = imp_vol(price_10[j, i], some_option_list[i])
        output_imp_vol_100[j, i] = imp_vol(price_100[j, i], some_option_list[i])
        output_imp_vol_1000[j, i] = imp_vol(price_1000[j, i], some_option_list[i])
        output_imp_vol_10000[j, i] = imp_vol(price_10000[j, i], some_option_list[i])
    j += 1

np.savetxt("Data/MC/Heston_mc_imp_vol_1.csv", output_imp_vol_1, delimiter=",")
np.savetxt("Data/MC/Heston_mc_imp_vol_10.csv", output_imp_vol_10, delimiter=",")
np.savetxt("Data/MC/Heston_mc_imp_vol_100.csv", output_imp_vol_100, delimiter=",")
np.savetxt("Data/MC/Heston_mc_imp_vol_1000.csv", output_imp_vol_1000, delimiter=",")
np.savetxt("Data/MC/Heston_mc_imp_vol_10000.csv", output_imp_vol_10000, delimiter=",")