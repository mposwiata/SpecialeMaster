import numpy as np
import time
from numba import jit
from scipy.stats import norm

@jit(nopython=True)
def monteCarloBS(mat, strike, sigma, r, spot, paths):
    var = sigma * sigma * mat
    vol = np.sqrt(var)
    movedSpot = spot * np.exp(r * mat - 0.5 * var)
    payoffSum = np.sum(np.maximum(movedSpot * np.exp(vol * np.random.standard_normal(paths)) - strike, 0))
    return np.exp(-r * mat) * payoffSum / paths

