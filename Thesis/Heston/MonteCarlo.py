import numpy as np
from scipy.stats import norm
from Thesis.Heston import HestonModel as hm, AndersenLake as al
from Thesis.misc import VanillaOptions as vo

def Heston_monte_carlo(some_model : hm.HestonClass, some_option : vo.VanillaOption, paths : int):
    dt = 252
    time_steps = int(some_option.tau * dt)
    forward_log = np.log(some_model.forward)
    vol = some_model.vol
    delta_t = some_option.tau / time_steps
    
    for i in range(time_steps):
        exp_part = np.exp(-some_model.kappa * delta_t)
        N_F = np.random.standard_normal(paths)
        N_v = some_model.rho * N_F + np.sqrt(1 - some_model.rho * some_model.rho) * np.random.standard_normal(paths)

        x = vol * exp_part + some_model.theta * (1 - exp_part)

        var = vol * some_model.epsilon * some_model.epsilon / some_model.kappa * (
            exp_part - np.exp(-2 * some_model.kappa * delta_t)
        ) + some_model.theta * some_model.epsilon * some_model.epsilon / (2 * some_model.kappa) * (
            1 - exp_part
        ) ** 2

        y = np.sqrt(np.log(var / (x * x) + 1))
        
        forward_log = forward_log - 0.5 * vol * delta_t + np.sqrt(np.maximum(vol, 0)) * np.sqrt(delta_t) * N_F

        vol = x * np.exp(- (y * y) / 2 + y * N_v)
    
    forward = np.exp(forward_log)

    return np.exp(-some_model.rate * some_option.tau) * (np.average(some_option(forward)))