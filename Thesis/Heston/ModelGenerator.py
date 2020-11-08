import numpy as np
import sys
import os
sys.path.append(os.getcwd()) # added for calc server support
from multiprocess import Pool, cpu_count

from Thesis.Heston import NNModelGenerator as mg
from Thesis.Heston import DataGeneration as dg

### Wide model
model_wide = dg.model_input_generator()
wide_imp = np.loadtxt("Data/hestonGridImpVol_wide.csv", delimiter = ",")

### Sobol, wider
sobol_input = np.loadtxt("Data/hestonSobolGridInput2_279936.csv", delimiter = ",")
sobol_imp = np.loadtxt("Data/hestonSobolGridImpVol2_279936.csv", delimiter = ",")

# Grid filtering, rows with 0 in
wide_imp_filter = np.all(wide_imp != 0, axis = 1)
model_wide = model_wide[wide_imp_filter, :]
wide_imp = wide_imp[wide_imp_filter, :]

sobol_imp_filter = np.all(sobol_imp != 0, axis = 1)
sobol_input = sobol_input[sobol_imp_filter, :]
sobol_imp = sobol_imp[sobol_imp_filter, :]

sobol_set = [
    [model_wide, wide_imp, 1, 500, "standard_1_500", False, "normal", "normalize"],
    [model_wide, wide_imp, 1, 1000, "standard_1_1000", False, "normal", "normalize"],
    [model_wide, wide_imp, 2, 500, "standard_2_500", False, "normal", "normalize"],
    [model_wide, wide_imp, 2, 1000, "standard_2_1000", False, "normal", "normalize"],
    [model_wide, wide_imp, 3, 500, "standard_3_500", False, "normal", "normalize"],
    [model_wide, wide_imp, 3, 1000, "standard_3_1000", False, "normal", "normalize"],
    [model_wide, wide_imp, 4, 500, "standard_4_500", False, "normal", "normalize"],
    [model_wide, wide_imp, 4, 1000, "standard_4_1000", False, "normal", "normalize"],
    [model_wide, wide_imp, 5, 500, "standard_5_500", False, "normal", "normalize"],
    [model_wide, wide_imp, 5, 1000, "standard_5_1000", False, "normal", "normalize"],
    [sobol_input, sobol_imp, 1, 500, "sobol_1_500", False, "normal", "normalize"],
    [sobol_input, sobol_imp, 1, 1000, "sobol_1_1000", False, "normal", "normalize"],
    [sobol_input, sobol_imp, 2, 500, "sobol_2_500", False, "normal", "normalize"],
    [sobol_input, sobol_imp, 2, 1000, "sobol_2_1000", False, "normal", "normalize"],
    [sobol_input, sobol_imp, 3, 500, "sobol_3_500", False, "normal", "normalize"],
    [sobol_input, sobol_imp, 3, 1000, "sobol_3_1000", False, "normal", "normalize"],
    [sobol_input, sobol_imp, 4, 500, "sobol_4_500", False, "normal", "normalize"],
    [sobol_input, sobol_imp, 4, 1000, "sobol_4_1000", False, "normal", "normalize"],
    [sobol_input, sobol_imp, 5, 500, "sobol_5_500", False, "normal", "normalize"],
    [sobol_input, sobol_imp, 5, 1000, "sobol_5_1000", False, "normal", "normalize"]
]

cpu_cores = min(cpu_count(), len(sobol_set))
# parallel
pool = Pool(cpu_cores)
res_sobol = pool.starmap(mg.NNModel, sobol_set)
print(res_sobol)
