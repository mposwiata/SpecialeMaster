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
    
    def BSDelta(self, option : vo.VanillaOption) -> float:
        d1 = (np.log(self.forward / option.strike) + 0.5 * self.vol * self.vol * option.tau) / (self.vol * np.sqrt(option.tau))
        return np.exp(-self.rate * option.tau) * ndtr(d1)

    def BSGamma(self, option : vo.VanillaOption) -> float:
        d1 = (np.log(self.forward / option.strike) + 0.5 * self.vol * self.vol * option.tau) / (self.vol * np.sqrt(option.tau))
        return np.exp(-self.rate * option.tau) * norm.pdf(d1) / (self. forward * self.vol * np.sqrt(option.tau))

    def BSVega(self, option : vo.VanillaOption) -> float:
        d1 = (np.log(self.forward / option.strike) + 0.5 * self.vol * self.vol * option.tau) / (self.vol * np.sqrt(option.tau))
        return np.exp(-self.rate * option.tau) * self.forward * norm.pdf(d1) * np.sqrt(option.tau)

    def impVol(self, price : float, option : vo.VanillaOption): 
        def root_func(vol : float, price : float, option : vo.VanillaOption) -> float:
            self.vol = vol
            return self.BSFormula(option) - price

        return root(root_func, 2, args = (price, option),tol=10e-6).x

    def delta2(self, option : vo.VanillaOption, a : float, b : float) -> float:
        d1 = (np.log(self.forward / option.strike) + 0.5 * self.vol * self.vol * option.tau) / (self.vol * np.sqrt(option.tau))
        d2 = d1 - self.vol * np.sqrt(option.tau)
        d1_2 = (self.vol - 2 * self.forward * np.log(self.forward / option.strike) * self.vol / (b - a)) / (np.sqrt(option.tau) * self.forward * self.vol**3)
        d2_2 = d1_2 - np.sqrt(option.tau) * self.vol / (b - a)

        return np.exp(-self.rate) * (self.forward * d1_2 * norm.pdf(d1) - option.strike * d2_2 * norm.pdf(d2) + ndtr(d1))

    def gamma2(self, option : vo.VanillaOption, a : float, b : float) -> float:
        def phi(x):
            return - x * np.exp(- x ** 2 / 2) / (2 * np.sqrt(np.pi))
        d1 = (np.log(self.forward / option.strike) + 0.5 * self.vol * self.vol * option.tau) / (self.vol * np.sqrt(option.tau))
        d2 = d1 - self.vol * np.sqrt(option.tau)
        d1_2 = (self.vol - 2 * self.forward * np.log(self.forward / option.strike) * self.vol / (b - a)) / (np.sqrt(option.tau) * self.forward * self.vol**3)
        d2_2 = d1_2 - np.sqrt(option.tau) * self.vol / (b - a)
        sx = self.vol
        sx1 = self.vol / (b - a)
        sx2 = self.vol / ((b - a) ** 2)
        x = self.forward
        k = option.strike
        t = option.tau
        d1_2_2 = (
            sx ** 3 * sx2 * t * x ** 2 + 4 * sx1 ** 2 * np.log(x/k) * x**2 - 2 * sx * sx2 * np.log(x/k)*x**2 - 4 * sx1 * x * sx - 2 * sx ** 2
        ) / (2 * np.sqrt(option.tau) * self.forward ** 2 * sx ** 3)
        d2_2_2 = d1_2_2 - np.sqrt(t) * sx2

        return np.exp(-self.rate * t) * (
            (
                2 * d1_2 * norm.pdf(d1) + self.forward * (d1_2 ** 2 * phi(d1) + d1_2_2 * norm.pdf(d1))
            ) - option.strike * (
                d2_2 ** 2 * phi(d2) + d2_2_2 * norm.pdf(d2)
            )
        )
