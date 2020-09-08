import numpy as np
from Thesis.Functions import hagan
from Thesis.Functions import black

def SABRDistBlack(f0, alpha, beta, strike, nu, rho, mat, type = "pdf"):
    epsilon = 1e-3 #setting low eps for central difference approximations
    strike_m = np.array([strike + 0.5*epsilon, strike, strike - 0.5*epsilon])
    imp_vol_m = hagan.hagan_sigma_b_v(f0, alpha, beta, strike_m, nu, rho, mat)
    prices_m = black.BlackPricer_v(f0, strike_m, mat, imp_vol_m)
    if (type == "pdf"):
        return np.array([1,-2,1]).dot(prices_m) / (0.25*epsilon*epsilon)
    else:
        return 1 + np.array([1,0,-1]).dot(prices_m) / epsilon

SABRDistBlack_v = np.vectorize(SABRDistBlack)