import numpy as np

def nu_par(mat, Ts):
    vol = np.linspace(start = 0.05, stop = 4, num = 10)
    return np.sqrt(Ts / mat) * vol

def sigma_temp(mat, eta_sigma, sigma_0, nu):
    return sigma_0 * np.exp(-0.5*nu*nu*mat + nu*eta_sigma*np.sqrt(mat))

def strike_par(mat, sigma_0, rho, nu):
    eta_sigma = 1.5
    eta = np.array([-3.5, 3.5])
    f0 = 1 # setting F(0,T) to 1
    return f0 * np.exp(
        -0.5*sigma_0*sigma_0/(nu*nu) * (np.exp(nu*nu)-1) +
        rho / nu * (
            sigma_temp(mat, eta_sigma, sigma_0, nu) - sigma_0
        ) + eta * sigma_0 / nu * np.power(
            np.exp(nu*nu*mat)-1, 0.5
        )
        )