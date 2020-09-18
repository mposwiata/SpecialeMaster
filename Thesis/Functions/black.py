# implementation of the Black swaption pricer

import numpy as np
import timeit
from scipy.stats import norm
import sys

def BlackPricer(f0, strike, mat, sigmaB):
    d1 = (np.log(f0/strike)+0.5*sigmaB*sigmaB*mat) / (sigmaB*np.sqrt(mat))
    d2 = (np.log(f0/strike)-0.5*sigmaB*sigmaB*mat) / (sigmaB*np.sqrt(mat))
    return f0 * norm.cdf(d1)-strike*norm.cdf(d2)

BlackPricer_v = np.vectorize(BlackPricer)