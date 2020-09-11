import numpy as np

def hagan_sigma_b(mat, alpha, rho, nu, strike, f0 = 1, beta = 1):
    if (np.abs(f0 - strike) < 1e-12):
        strike = f0
    elif (strike < 1e-12):
        return 0
    power2 = (1-beta)*(1-beta)
    if strike != f0:
        power = (1-beta)/2
        epsilon = nu / alpha * np.power(f0 * strike, power) * np.log(f0 / strike)
        x = np.log(
            (np.sqrt(1-2*rho*epsilon+epsilon*epsilon)-rho+epsilon) / (1-rho)
        )
        sigma = alpha / (np.power(f0 * strike, power)) * 1 / (
            1 + power2 / 24 * np.power(np.log(f0 / strike), 2) + power2 * power2 / 1920 * np.power(np.log(f0 / strike), 4)
        ) * epsilon / x * (
            1 + (
                power2 * alpha * alpha / (24 * np.power(f0 * strike, 1-beta)) + rho * alpha * nu * beta / (4 * np.power(f0 * strike, (1-beta)/2)) + (2-3*rho*rho)/24 * nu * nu
            ) * mat
        )
        return sigma
    else:
        sigma = alpha / np.power(f0, 1-beta) * (
            1 + (
                power2 * alpha * alpha / (24 * np.power(f0, 2-2*beta)) + rho * alpha * nu * beta / (4 * np.power(f0, 1-beta)) + (2-3*rho*rho)/24 * nu * nu
            ) * mat
        )
        return sigma

hagan_sigma_b_v = np.vectorize(hagan_sigma_b)