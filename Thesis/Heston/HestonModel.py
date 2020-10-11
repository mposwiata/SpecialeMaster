#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:46:56 2020

@author: Martin
"""

import numpy as np
from scipy.stats import norm

# Own project
from Thesis.Heston import helper
from Thesis.BlackScholes import BlackScholes as bs
from Thesis.misc import VanillaOptions as vo

NUMPY_COMPLEX128_MAX = np.finfo(np.complex128).max
NUMPY_LOG_COMPLEX128_MAX = np.log(NUMPY_COMPLEX128_MAX)

class HestonClass:
    def __init__(self, forward : float, vol : float, kappa : float, theta : float, epsilon : float, rho : float, rate : float):
        self.forward = forward
        self.vol = vol
        self.kappa = kappa
        self.theta = theta
        self.epsilon = epsilon
        self.rho = rho
        self.rate = rate
        self.omega = None
        self.phi = None

    def charFunc(self, u : complex, tau : float) -> complex:
        beta = self.kappa - 1J * self.epsilon * self.rho * u
        D = np.sqrt(beta * beta + self.epsilon * self.epsilon * u * (u + 1J))

        if beta.real * D.real + beta.imag * D.imag > 0:
            r = - self.epsilon * self.epsilon * u * (u + 1J) / (beta + D)
        else:
            r = beta - D

        if D != 0:
            y = np.expm1(- D * tau) / (2 * D)
        else:
            y = - tau / 2

        A = self.kappa * self.theta / (self.epsilon * self.epsilon) * (
            r * tau - 2 * np.log1p(-r * y)
        )

        B = u * (u + 1J) * y / (1 - r * y)

        
        if A + B * self.vol > NUMPY_LOG_COMPLEX128_MAX:
            raise OverflowError("too large exponent in characteristic function")
        

        return np.exp(A + B * self.vol)

    def logCharFunc(self, alpha : float, tau : float) -> float:
        u = -1J*(1+alpha)
        beta = (self.kappa - 1J * self.epsilon * self.rho * u).real
        D2 = (beta * beta + self.epsilon * self.epsilon * u * (u + 1J)).real

        if D2 > 0:
            D = np.sqrt(D2)
            dt2 = D * tau / 2
            A = self.kappa * self.theta / (self.epsilon * self.epsilon) * (
                beta * tau - np.log(
                    (np.cosh(dt2) + beta * np.sinh(dt2) / D) ** 2
                )
            )

            B = alpha * (1 + alpha) * (np.sinh(dt2) / D) / (
                np.cosh(dt2) + beta * np.sinh(dt2) / D
            )
        else:
            D = np.sqrt(-D2)
            dt2 = D * tau / 2
            A = self.kappa * self.theta / (self.epsilon * self.epsilon) * (
                beta * tau - np.log(
                    (np.cos(dt2) + beta * np.sin(dt2) / D) ** 2
                )
            )

            B = alpha * (1 + alpha) * (np.sin(dt2) / D) / (
                np.cos(dt2) + beta * np.sin(dt2) / D
            )

        return A + B * self.vol

    def AndersenLakeParameters(self, option : vo.VanillaOption):
        self.omega = np.log(self.forward / option.strike)
        r = self.rho - (self.epsilon * self.omega) / (self.vol + self.kappa * self.theta * option.tau)
        if r * self.omega < 0:
            self.phi = np.pi / 12 * np.sign(self.omega)
        else:
            self.phi = 0 
    
    def impVol(self, price : float, option : vo.VanillaOption):
        tempBS = bs.BlackScholesForward(self.forward, self.vol, self.rate)

        return tempBS.impVol(price, option)