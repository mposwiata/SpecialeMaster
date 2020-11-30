import numpy as np

from Thesis.BlackScholes import BlackScholes as bs
from Thesis.misc import VanillaOptions as vo

def Black_monte_carlo(BS_model : bs.BlackScholesForward, some_option : vo.VanillaOption, paths : int):
    log_F_T = np.log(BS_model.forward) - 0.5 * BS_model.vol * BS_model.vol * some_option.tau + BS_model.vol * np.sqrt(some_option.tau) * np.random.standard_normal(paths)
    forward = np.exp(log_F_T)

    return np.exp(-BS_model.rate * some_option.tau) * (np.average(some_option(forward)))