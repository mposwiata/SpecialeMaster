import numpy as np

from Thesis.Heston import DataGeneration as dg, HestonModel as hm, AndersenLake as al
from Thesis.misc import VanillaOptions as vo
from Thesis.BlackScholes import BlackScholes as bs

option_input = dg.option_input_generator()
some_option_list = np.array([])
for option in option_input:
    some_option_list = np.append(some_option_list, vo.EUCall(option[0], option[1]))
option_list = some_option_list

input_array = np.array([100.   ,   0.105,   1.05 ,   0.105,   1.05 ,   0.   ,   0.1  ])

some_model = hm.HestonClass(input_array[0], input_array[1], input_array[2], input_array[3], input_array[4], input_array[5], input_array[6])
output_lenght = np.shape(option_list)[0]
output_price = np.zeros(output_lenght, dtype=np.float64)
output_imp_vol = np.zeros(output_lenght, dtype=np.float64)

i = 0
output_price[i] = al.Andersen_Lake(some_model, option_list[i])
output_imp_vol[i] = some_model.impVol(output_price[i], option_list[i])
if output_imp_vol[i] == -1:
    print(vars(some_model))
    print(output_price[i])
    print(vars(option_list[i]))

some_option = vo.EUCall(0.01, 75)
imp_vol = 0.38549383
some_bs_model = bs.BlackScholesForward(100, imp_vol, 0.1)
print(some_bs_model.BSFormula(some_option, imp_vol))
print(some_bs_model.impVol(output_price[i], option_list[i]))

some_bs_model.impVol(24.975012, some_option)

print(some_model.impVol(output_price[i], option_list[i]))