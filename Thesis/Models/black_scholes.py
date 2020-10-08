#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:46:56 2020

@author: Martin
"""

import numpy as np
from scipy.stats import norm

def BlackScholesFormula(spot : float, strike : float, mat : float, sigma : float, r : float, isCall : bool = True) -> float:
    d1 = 1 / (sigma * np.sqrt(mat)) * (np.log(spot / strike) + (r + (sigma * sigma)/2)*(mat))
    d2 = d1 - sigma * np.sqrt(mat)
    if isCall:
        return norm.cdf(d1) * spot - norm.cdf(d2) * strike * np.exp(-r * mat)
    else:
       return norm.cdf(-d2)*strike*np.exp(-r*mat)-norm.cdf(-d1)*spot

vBlackScholesFormula = np.vectorize(BlackScholesFormula)

def BlackScholesDelta(spot : float, strike : float, mat : float, sigma : float, r : float, isCall : bool = True) -> float:
    d1 = 1 / (sigma * np.sqrt(mat)) * (np.log(spot / strike) + (r + (sigma * sigma)/2)*(mat))
    if isCall:
        return norm.cdf(d1)
    else:
        return norm.cdf(-d1)

vBlackScholesDelta = np.vectorize(BlackScholesDelta)

def BlackScholesGamma(spot : float, strike : float, mat : float, sigma : float, r : float) -> float:
    d1 = 1 / (sigma * np.sqrt(mat)) * (np.log(spot / strike) + (r + (sigma * sigma)/2)*(mat))
    return norm.pdf(d1) / (spot * sigma * np.sqrt(mat))

vBlackScholesGamma = np.vectorize(BlackScholesGamma)    

class BlackScholesClass:
    def __init__(self, spot, mat, strike, sigma, r):
        self.spot = spot
        self.mat = mat
        self.strike = strike
        self.sigma = sigma
        self.r = r
        
    def call(self):
        return BlackScholesFormula(self.spot, self.mat, self.strike, self.sigma, self.r, "call")
    
    def put(self):
        return BlackScholesFormula(self.spot, self.mat, self.strike, self.sigma, self.r, "put")

class BlackScholesForwardClass:
    def __init__(self, spot, vol, rate):
        self.spot = spot
        self.vol = vol
        self.rate = rate
