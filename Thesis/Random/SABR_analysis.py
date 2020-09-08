import numpy as np
from Thesis.Functions import SABR
from Thesis.Functions import black
from Thesis.Functions import hagan
import matplotlib.pyplot as plt

strike = np.arange(0.1, 2.5, 0.02) * 1e-2
f0 = 0.01 # 1%
alpha = 0.005 # 50bps
beta = 0.25 
rho = -0.1
nu = 0.5
mat = 10 
epsilon = 1e-3

imp_vol = hagan.hagan_sigma_b_v(f0, alpha, beta, strike, nu, rho, mat)

densityBlack = SABR.SABRDistBlack_v(f0, alpha, beta, strike, nu, rho, mat)

densityBlackCdf = SABR.SABRDistBlack_v(f0, alpha, beta, strike, nu, rho, mat, "cdf")

plt.plot(strike, densityBlackCdf)
plt.savefig("SabrCdf.jpeg")
