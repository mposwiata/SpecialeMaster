import numpy as np
import time
from scipy.stats import norm
from Thesis.misc import VanillaOptions as vo
from scipy.special import ndtr

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

    def impVol(self, price, option : vo.VanillaOption):
        precision = 10e-6
        max_iter = 200

        for i in range(max_iter):
            BSPrice = self.BSFormula(option)
            vega = self.BSVega(option)
            diff = price - BSPrice
            if (abs(diff) < precision):
                return self.vol
            self.vol = self.vol + diff / vega
        
        return self.vol