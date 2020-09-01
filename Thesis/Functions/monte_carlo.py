import numpy as np
import time
from scipy.stats import norm

def monteCarloBS(spot, mat, strike, sigma, r, paths):
    var = sigma * sigma * mat
    vol = np.sqrt(var)
    movedSpot = spot * np.exp(r * mat - 0.5 * var)
    payoffSum = np.sum(np.maximum(movedSpot * np.exp(vol * np.random.standard_normal(paths)) - strike, 0))
    return np.exp(-r * mat) * payoffSum / paths
