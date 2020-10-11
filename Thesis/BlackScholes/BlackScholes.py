import numpy as np
import time
from scipy.stats import norm
from Thesis.misc import VanillaOptions as vo
from scipy.special import ndtr
from scipy.optimize import root

class BlackScholesForward:
    def __init__(self, forward : float, vol : float, rate : float):
        self.forward = forward
        self.vol = vol
        self.rate = rate

    def BSFormula(self, option : vo.VanillaOption) -> float:
        d1 = (np.log(self.forward / option.strike) + 0.5 * self.vol * self.vol * option.tau) / (self.vol * np.sqrt(option.tau))
        d2 = d1 - self.vol * np.sqrt(option.tau)

        return np.exp(-self.rate * option.tau) * (self.forward * ndtr(d1) - option.strike * ndtr(d2))
    
    def BSVega(self, option : vo.VanillaOption) -> float:
        d1 = (np.log(self.forward / option.strike) + 0.5 * self.vol * self.vol * option.tau) / (self.vol * np.sqrt(option.tau))
        return self.forward * norm._pdf(d1) * np.sqrt(option.tau)

    def impVol(self, price : float, option : vo.VanillaOption): 
        def root_func(vol : float, price : float, option : vo.VanillaOption) -> float:
            self.vol = vol
            return self.BSFormula(option) - price

        return root(root_func, 0.1, args = (price, option),tol=10e-6).x