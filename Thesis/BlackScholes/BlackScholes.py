import numpy as np
import time
from scipy.stats import norm
from Thesis.misc import VanillaOptions as vo
from scipy.special import ndtr
from scipy.optimize import root, brentq

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
        def root_func(some_vol : float, price : float, option : vo.VanillaOption) -> float:
            self.vol = some_vol
            if some_vol < -1:
                return -1
            return self.BSFormula(option) - price
        root_result = root(root_func, 0.4, args = (price, option), tol=10e-5)
        if root_result.success:
            return root_result.x
        else:
            return -1

    def total_delta(self, option : vo.VanillaOption, sigma_grad : float) -> float:
        return self.BSDelta(option) + self.BSVega(option) * sigma_grad

    def total_gamma(self, option : vo.VanillaOption, sigma_grad : float, sigma_grad2 : float) -> float:
        def phi(x):
            return - x * np.exp(- x ** 2 / 2) / (2 * np.sqrt(np.pi))
        d1 = (np.log(self.forward / option.strike) + 0.5 * self.vol * self.vol * option.tau) / (self.vol * np.sqrt(option.tau))
        d1_grad = option.tau / np.sqrt(option.tau) - d1 / self.vol
        vega_grad = np.exp(-self.rate * option.tau) * self.forward * phi(d1) * d1_grad
        delta_grad = np.exp(-self.rate * option.tau) * norm.pdf(d1) * d1_grad

        return self.BSGamma(option) + vega_grad * sigma_grad * sigma_grad + self.BSVega(option) * sigma_grad2 + 2 * delta_grad * sigma_grad

    def delta2(self, option : vo.VanillaOption, a : float, b : float) -> float:
        x = self.forward
        k = option.strike
        sigma = self.vol
        t = option.tau
        d1 = (
            np.log(x/k) + sigma ** 2 * t / 2
        ) / (sigma * np.sqrt(t))
        d1_dx = (
            2 * np.log(x / k) * x + 2 * (a - b) - sigma ** 2 * t * x
        ) / (
            2 * sigma * (a - b) * x * np.sqrt(t)
        )
        d2 = d1 - sigma * np.sqrt(t)
        d2_dx = d1_dx - np.sqrt(t) * sigma / (b - a)

        return np.exp(-self.rate) * (self.forward * d1_dx * norm.pdf(d1) - option.strike * d2_dx * norm.pdf(d2) + ndtr(d1))

    def delta_grads(self, option : vo.VanillaOption, sigma_dx) -> float:
        x = self.forward
        k = option.strike
        sigma = self.vol
        #sigma_dx = grad / (b - a)
        t = option.tau
        d1 = (
            np.log(x/k) + sigma ** 2 * t / 2
        ) / (sigma * np.sqrt(t))
        d1_dx = (
            sigma ** 2 * sigma_dx * t * x - 2 * sigma_dx * np.log(x/k) * x + 2 * sigma
        ) / (
            2 * x * sigma ** 2 * np.sqrt(t)
        )
        d2 = d1 - sigma * np.sqrt(t)
        d2_dx = d1_dx - np.sqrt(t) * sigma_dx

        return np.exp(-self.rate) * (self.forward * d1_dx * norm.pdf(d1) - option.strike * d2_dx * norm.pdf(d2) + ndtr(d1))

    def gamma2(self, option : vo.VanillaOption, a : float, b : float) -> float:
        def phi(x):
            return - x * np.exp(- x ** 2 / 2) / (2 * np.sqrt(np.pi))
        x = self.forward
        k = option.strike
        sigma = self.vol
        t = option.tau
        d1 = (
            np.log(x/k) + sigma ** 2 * t / 2
        ) / (sigma * np.sqrt(t))
        d1_dx = (
            2 * np.log(x / k) * x + 2 * (a - b) - sigma ** 2 * t * x
        ) / (
            2 * sigma * (a - b) * x * np.sqrt(t)
        )
        d2 = d1 - sigma * np.sqrt(t)
        d2_dx = d1_dx - np.sqrt(t) * sigma / (b - a)

        d1_dx2 = - (
            np.log(x/k) * x ** 2 * sigma / (b - a) + x * (
                sigma **2 / 2 * t * x + a - b
            ) * sigma / (b - a) + sigma * (a - b - x)
        ) / (np.sqrt(t) * sigma ** 2 * (a - b) * x**2)

        d2_dx2 = d1_dx2 - np.sqrt(t) * sigma / ((b - a) ** 2)

        return np.exp(-self.rate * t) * (
            (
                2 * d1_dx * norm.pdf(d1) + self.forward * (d1_dx ** 2 * phi(d1) + d1_dx2 * norm.pdf(d1))
            ) - option.strike * (
                d2_dx ** 2 * phi(d2) + d2_dx2 * norm.pdf(d2)
            )
        )

    def gamma_grads(self, option : vo.VanillaOption, sigma_dx : float, sigma_dx2 : float) -> float:
        def phi(x):
            return - x * np.exp(- x ** 2 / 2) / (2 * np.sqrt(np.pi))
        x = self.forward
        k = option.strike
        sigma = self.vol
        #sigma_dx = grad / (b - a)
        #sigma_dx2 = grad2 / ((b - a) ** 2)
        t = option.tau
        d1 = (
            np.log(x/k) + sigma ** 2 * t / 2
        ) / (sigma * np.sqrt(t))
        d1_dx = (
            sigma ** 2 * sigma_dx * t * x - 2 * sigma_dx * np.log(x/k) * x + 2 * sigma
        ) / (
            2 * x * sigma ** 2 * np.sqrt(t)
        )
        d2 = d1 - sigma * np.sqrt(t)
        d2_dx = d1_dx - np.sqrt(t) * sigma_dx

        d1_dx2 = (
            sigma ** 3 * sigma_dx2 * t * x**2 - 2 * sigma * np.log(x/k) * sigma_dx2 * x**2 + \
                4 * sigma_dx ** 2 * np.log(x/k) * x**2 - 4 * sigma * sigma_dx * x - 2 * sigma ** 2
        ) / (
            2 * x**2 * sigma**3 * np.sqrt(t)
        )

        d2_dx2 = d1_dx2 - np.sqrt(t) * sigma_dx2

        return np.exp(-self.rate * t) * (
            (
                2 * d1_dx * norm.pdf(d1) + self.forward * (d1_dx ** 2 * phi(d1) + d1_dx2 * norm.pdf(d1))
            ) - option.strike * (
                d2_dx ** 2 * phi(d2) + d2_dx2 * norm.pdf(d2)
            )
        )
