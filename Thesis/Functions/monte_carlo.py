import numpy as np
import time
from scipy.stats import norm

def monteCarloBS(mat, strike, sigma, r, spot):
    paths = 1000
    var = sigma * sigma * mat
    vol = np.sqrt(var)
    moved_spot = spot * np.exp(r * mat - 0.5 * var)
    payoff_sum = np.sum(np.maximum(moved_spot * np.exp(vol * np.random.standard_normal(paths)) - strike, 0))
    return np.exp(-r * mat) * payoff_sum / paths

