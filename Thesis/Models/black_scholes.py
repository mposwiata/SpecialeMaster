#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:46:56 2020

@author: Martin
"""

import numpy as np
from scipy.stats import norm

def BlackScholesFormula(spot, strike, mat, sigma, r, type = "call"):
    d1 = 1 / (sigma * np.sqrt(mat)) * (np.log(spot / strike) + (r + (sigma * sigma)/2)*(mat))
    d2 = d1 - sigma * np.sqrt(mat)
    if type == "call":
        return norm.cdf(d1) * spot - norm.cdf(d2) * strike * np.exp(-r * mat)
    else:
       return norm.cdf(-d2)*strike*np.exp(-r*mat)-norm.cdf(-d1)*spot

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
