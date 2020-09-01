#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:46:56 2020

@author: Martin
"""

import numpy as np
from scipy.stats import norm

def BlackScholesFormula(spot, t, mat, strike, sigma, r, type = "call"):
    d1 = 1 / (sigma * np.sqrt(mat - t)) * (np.log(spot / strike) + (r + (sigma * sigma)/2)*(mat - t))
    d2 = d1 - sigma * np.sqrt(mat - t)
    if type == "call":
        return norm.cdf(d1) * spot - norm.cdf(d2) * strike * np.exp(-r * (mat-t))
    else:
       return norm.cdf(-d2)*strike*np.exp(-r*(mat-t))-norm.cdf(-d1)*spot

class BlackScholesClass:
    def __init__(self, spot, t, mat, strike, sigma, r):
        self.spot = spot
        self.t = t
        self.mat = mat
        self.strike = strike
        self.sigma = sigma
        self.r = r
        
    def call(self):
        return BlackScholesFormula(self.spot, self.t, self.mat, self.strike, self.sigma, self.r, "call")
    
    def put(self):
        return BlackScholesFormula(self.spot, self.t, self.mat, self.strike, self.sigma, self.r, "put")
