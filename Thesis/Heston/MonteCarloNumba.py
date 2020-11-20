import numpy as np
from scipy.stats import norm
from Thesis.Heston import HestonModel as hm, AndersenLake as al
from Thesis.misc import VanillaOptions as vo
from numba import jit

@jit(nopython=True)
def Heston_monte_carlo(model_array : np.ndarray, option_array, paths : int):
    forward = model_array[0]
    vol = model_array[1]
    kappa = model_array[2]
    theta = model_array[3]
    epsilon = model_array[4]
    rho = model_array[5]
    rate = model_array[6]

    tau = option_array[0]
    strike = option_array[1]
    
    dt = 252
    time_steps = int(tau * dt)
    forward_log = np.log(forward)
    delta_t = tau / time_steps
    
    for i in range(time_steps):
        exp_part = np.exp(-kappa * delta_t)
        N_F = np.random.standard_normal(paths)
        N_v = rho * N_F + np.sqrt(1 - rho * rho) * np.random.standard_normal(paths)

        x = vol * exp_part + theta * (1 - exp_part)

        var = vol * epsilon * epsilon / kappa * (
            exp_part - np.exp(-2 * kappa * delta_t)
        ) + theta * epsilon * epsilon / (2 * kappa) * (
            1 - exp_part
        ) ** 2

        y = np.sqrt(np.log(var / (x * x) + 1))
        

        forward_log = forward_log - 0.5 * vol * delta_t + np.sqrt(np.maximum(vol, 0)) * np.sqrt(delta_t) * N_F

        vol = x * np.exp(- (y * y) / 2 + y * N_v)
    
    forward = np.exp(forward_log)
    
    return np.exp(-rate * tau) * (np.average(np.maximum(forward - strike, 0)))