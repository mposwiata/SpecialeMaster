import numpy as np
import time
from scipy.optimize import fminbound, brentq
from scipy.integrate import quad

# Own project
from Thesis.misc import VanillaOptions as vo
from Thesis.Heston import HestonModel

def alpha_min_func(alpha : float, model : HestonModel.HestonClass, option: vo.VanillaOption) -> (float, float):
    return model.logCharFunc(alpha, option.tau) - np.log(alpha * (alpha + 1)) + alpha * model.omega

def optimal_alpha(model : HestonModel.HestonClass, option : vo.VanillaOption) -> float:
    epsilon = np.sqrt(np.finfo(float).eps) # used for open / closed intervals, as brentq uses closed intervals
    alpha_min, alpha_max = alpha_min_max(model, option)

    if model.omega >= 0:
        alpha, value = fminbound(alpha_min_func, x1 = alpha_min, x2 = -1 - epsilon, args = (model, option), full_output=True)[0:2]
    elif model.omega < 0 and model.kappa - model.rho * model.epsilon > 0:
        alpha, value = fminbound(alpha_min_func, x1 = epsilon, x2 = alpha_max, args = (model, option), full_output=True)[0:2]
    else:
        alpha, value = fminbound(alpha_min_func, x1 = epsilon, x2 = alpha_max, args = (model, option), full_output=True)[0:2]
        if value > 9:
            alpha = fminbound(alpha_min_func, x1 = alpha_min, x2 = -1 - epsilon, args = (model, option), full_output=True)[0:2]

    return alpha

def k_p_m( x : float, model : HestonModel.HestonClass, option : vo.VanillaOption) -> (float, float):
    first_term = model.epsilon - 2 * model.rho * model.kappa
    sqrt_term = (model.epsilon - 2 * model.rho * model.kappa) ** 2 + 4 * (model.kappa * model.kappa + x * x / (option.tau * option.tau)) * (1 - model.rho * model.rho)

    denominator = 2 * model.epsilon * (1 - model.rho * model.rho)

    return ((first_term - np.sqrt(sqrt_term)) / denominator, (first_term + np.sqrt(sqrt_term)) / denominator)

def critical_moments(k : float, model : HestonModel.HestonClass, option : vo.VanillaOption) -> float:
    epsilon = np.sqrt(np.finfo(float).eps)
    k_minus, k_plus = k_p_m(0, model, option)
    beta = model.kappa - model.rho * model.epsilon * k
    
    D = np.sqrt(beta**2 + model.epsilon**2 * (-1j * k) * ((-1j * k) + 1j))

    if k < k_minus or k > k_plus:
        D = abs(D)
        return np.cos(D * option.tau / 2) + beta * np.sin(D * option.tau / 2) / D
    else:
        D = D.real
        if D == 0: # Precission sets very low numbers to 0.0, will destroy the folowwing 
            D = epsilon
        return np.cosh(D * option.tau / 2) + beta * np.sinh(D * option.tau / 2) / D

def alpha_min_max(model : HestonModel.HestonClass, option : vo.VanillaOption) -> (float, float):
    epsilon = np.sqrt(np.finfo(float).eps) # used for open / closed intervals, as brentq uses closed intervals

    k_minus, k_plus = k_p_m(0, model, option)

    k_minus_pi, k_plus_pi = k_p_m(np.pi, model, option)

    k_minus_2pi, k_plus_2pi = k_p_m(2 * np.pi, model, option)

    # Finding k_
    k_min = brentq(critical_moments, a=k_minus_2pi + epsilon, b=k_minus - epsilon, args=(model, option))
    #k_min = brentq(critical_moments, a=k_minus_2pi, b=k_minus, args=(model, option))

    kappa_rho_epsilon = model.kappa - model.rho * model.epsilon

    # Finding k+
    if kappa_rho_epsilon > 0:
        lower_bound, upper_bound = k_plus + epsilon, k_plus_2pi - epsilon
    elif kappa_rho_epsilon < 0:
        T = - 2 / (model.kappa - model.rho * model.epsilon * k_plus)
        if option.tau < T:
            lower_bound, upper_bound = k_plus + epsilon, k_plus_pi - epsilon
        else:
            lower_bound, upper_bound = 1 + epsilon, k_plus 
    else:
        lower_bound, upper_bound = 1 + epsilon, k_plus_pi - epsilon
    
    k_max = brentq(critical_moments, a=lower_bound, b=upper_bound, args=(model, option))
    
    return k_min - 1, k_max - 1

def Q_H(z, model : HestonModel.HestonClass, option : vo.VanillaOption) -> complex:
    return model.charFunc(z - 1J, option.tau) / (z * (z - 1J))

def Andersen_Lake(model : HestonModel.HestonClass, option : vo.VanillaOption) -> float:
    # calculating andersen lake parameters
    model.AndersenLakeParameters(option)
    alpha = optimal_alpha(model, option)
    
    def integrand(x):
        return (np.exp(-x * np.tan(model.phi) * model.omega + 1J * x * model.omega) * Q_H(-1J * alpha + x * (1 + 1J * np.tan(model.phi)), model, option) * (1 + 1J * np.tan(model.phi))).real
    
    integral = np.exp(alpha * model.omega) * quad(integrand, 0, np.inf)[0]
    
    R = model.forward * (alpha <= 0) - option.strike * (alpha <= -1) - 0.5 * (model.forward * (alpha == 0) - option.strike * (alpha == -1))

    return np.exp(-model.rate * option.tau) * (R - model.forward / np.pi * integral)
