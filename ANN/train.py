import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

def BS_Call(S,K,r,T,v,d=0):
    d1 = (np.log(float(S)/K)+((r-d)+v*v/2.)*T)/(v*np.sqrt(T))
    d2 = d1-v*np.sqrt(T)
    return S*np.exp(-d*T)*ss.norm.cdf(d1)-K*np.exp(-r*T)*ss.norm.cdf(d2)

# Stock Price
S = np.arange(10, 200, 2)

# Strike Price
# To avoid extreme prices - K  will be a multiple of the Stock Price (S) rather than a completely seperate RV
K = np.random.random(len(S)) + 0.5

# Interest Rate
r = 0.05

# Time
T = np.arange(0.1, 1, 0.10)

# Volatility
V = np.arange(0.1, 0.6, 0.05)

# Number of option prices = life begins at a million examples...
no_of_options = len(S)*len(K)*len(T)*len(V)

# Create numpy array to store option data
prices = np.empty([no_of_options,5], dtype=float)

# Track time record (1-2 minutes)

# Loop through parameters
x = 0
for s in S:
    for k in K:
        for t in T:
            for v in V:
                prices[x,:] = [s,s*k,t,v,BS_Call(s,s*k,r,t,v)]
                x+=1

#Do not store constant interest rate value
option_df = pd.DataFrame(index = range(no_of_options), columns = 
                         ['Stock', 
                          'Strike',
                          'Time',
                          'Volatility',
                          'Call Price'], data = prices )

option_df.to_csv("test.csv")