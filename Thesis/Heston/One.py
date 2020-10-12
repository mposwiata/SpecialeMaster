import numpy as np
import time
import itertools
from multiprocess import Pool
from scipy.stats import norm
from scipy.special import ndtr
from scipy.optimize import root
from scipy.optimize import fminbound, brentq
from scipy.integrate import quad

NUMPY_COMPLEX128_MAX = np.finfo(np.complex128).max
NUMPY_LOG_COMPLEX128_MAX = np.log(NUMPY_COMPLEX128_MAX)

class VanillaOption:
    def __init__(self, tau, strike):
        self.tau = tau
        self.strike = strike

class EUCall(VanillaOption):
    def __init__(self, tau, strike):
        VanillaOption.__init__(self, tau, strike)

    def __call__(self, forward):
        return np.maximum(forward - self.strike, 0)

class EUPut(VanillaOption):
    def __init__(self, tau, strike):
        VanillaOption.__init__(self, tau, strike)

    def __call__(self, forward):
        return np.maximum(self.strike - forward, 0)

class BlackScholesForward:
    def __init__(self, forward : float, vol : float, rate : float):
        self.forward = forward
        self.vol = vol
        self.rate = rate

    def BSFormula(self, option : VanillaOption) -> float:
        d1 = (np.log(self.forward / option.strike) + 0.5 * self.vol * self.vol * option.tau) / (self.vol * np.sqrt(option.tau))
        d2 = d1 - self.vol * np.sqrt(option.tau)

        return np.exp(-self.rate * option.tau) * (self.forward * ndtr(d1) - option.strike * ndtr(d2))
    
    def BSVega(self, option : VanillaOption) -> float:
        d1 = (np.log(self.forward / option.strike) + 0.5 * self.vol * self.vol * option.tau) / (self.vol * np.sqrt(option.tau))
        return self.forward * norm._pdf(d1) * np.sqrt(option.tau)

    def impVol(self, price : float, option : VanillaOption): 
        def root_func(vol : float, price : float, option : VanillaOption) -> float:
            self.vol = vol
            return self.BSFormula(option) - price

        return root(root_func, 0.1, args = (price, option),tol=10e-6).x

class HestonClass:
    def __init__(self, forward : float, vol : float, kappa : float, theta : float, epsilon : float, rho : float, rate : float):
        self.forward = forward
        self.vol = vol
        self.kappa = kappa
        self.theta = theta
        self.epsilon = epsilon
        self.rho = rho
        self.rate = rate
        self.omega = None
        self.phi = None

    def charFunc(self, u : complex, tau : float) -> complex:
        beta = self.kappa - 1J * self.epsilon * self.rho * u
        D = np.sqrt(beta * beta + self.epsilon * self.epsilon * u * (u + 1J))

        if beta.real * D.real + beta.imag * D.imag > 0:
            r = - self.epsilon * self.epsilon * u * (u + 1J) / (beta + D)
        else:
            r = beta - D

        if D != 0:
            y = np.expm1(- D * tau) / (2 * D)
        else:
            y = - tau / 2

        A = self.kappa * self.theta / (self.epsilon * self.epsilon) * (
            r * tau - 2 * np.log1p(-r * y)
        )

        B = u * (u + 1J) * y / (1 - r * y)

        
        if A + B * self.vol > NUMPY_LOG_COMPLEX128_MAX:
            raise OverflowError("too large exponent in characteristic function")
        

        return np.exp(A + B * self.vol)

    def logCharFunc(self, alpha : float, tau : float) -> float:
        u = -1J*(1+alpha)
        beta = (self.kappa - 1J * self.epsilon * self.rho * u).real
        D2 = (beta * beta + self.epsilon * self.epsilon * u * (u + 1J)).real

        if D2 > 0:
            D = np.sqrt(D2)
            dt2 = D * tau / 2
            A = self.kappa * self.theta / (self.epsilon * self.epsilon) * (
                beta * tau - np.log(
                    (np.cosh(dt2) + beta * np.sinh(dt2) / D) ** 2
                )
            )

            B = alpha * (1 + alpha) * (np.sinh(dt2) / D) / (
                np.cosh(dt2) + beta * np.sinh(dt2) / D
            )
        else:
            D = np.sqrt(-D2)
            dt2 = D * tau / 2
            A = self.kappa * self.theta / (self.epsilon * self.epsilon) * (
                beta * tau - np.log(
                    (np.cos(dt2) + beta * np.sin(dt2) / D) ** 2
                )
            )

            B = alpha * (1 + alpha) * (np.sin(dt2) / D) / (
                np.cos(dt2) + beta * np.sin(dt2) / D
            )

        return A + B * self.vol

    def AndersenLakeParameters(self, option : VanillaOption):
        self.omega = np.log(self.forward / option.strike)
        r = self.rho - (self.epsilon * self.omega) / (self.vol + self.kappa * self.theta * option.tau)
        if r * self.omega < 0:
            self.phi = np.pi / 12 * np.sign(self.omega)
        else:
            self.phi = 0 
    
    def impVol(self, price : float, option : VanillaOption):
        tempBS = BlackScholesForward(self.forward, self.vol, self.rate)

        return tempBS.impVol(price, option)

def alpha_min_func(alpha : float, model : HestonClass, option: VanillaOption) -> (float, float):
    return model.logCharFunc(alpha, option.tau) - np.log(alpha * (alpha + 1)) + alpha * model.omega

def optimal_alpha(model : HestonClass, option : VanillaOption) -> float:
    epsilon = np.finfo(float).eps # used for open / closed intervals, as brentq uses closed intervals
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

def k_p_m( x : float, model : HestonClass, option : VanillaOption) -> (float, float):
    first_term = model.epsilon - 2 * model.rho * model.kappa
    sqrt_term = (model.epsilon - 2 * model.rho * model.kappa) ** 2 + 4 * (model.kappa * model.kappa + x * x / (option.tau * option.tau)) * (1 - model.rho * model.rho)

    denominator = 2 * model.epsilon * (1 - model.rho * model.rho)

    return ((first_term - np.sqrt(sqrt_term)) / denominator, (first_term + np.sqrt(sqrt_term)) / denominator)

def critical_moments(k :float, model : HestonClass, option : VanillaOption) -> float:
    epsilon = np.finfo(float).eps
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

def alpha_min_max(model : HestonClass, option : VanillaOption) -> (float, float):
    epsilon = np.finfo(float).eps # used for open / closed intervals, as brentq uses closed intervals

    k_minus, k_plus = k_p_m(0, model, option)

    k_minus_pi, k_plus_pi = k_p_m(np.pi, model, option)

    k_minus_2pi, k_plus_2pi = k_p_m(2 * np.pi, model, option)

    # Finding k_
    k_min = brentq(critical_moments, a=k_minus_2pi + epsilon, b=k_minus - epsilon, args=(model, option))

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

def Q_H(z, model : HestonClass, option : VanillaOption) -> complex:
    return model.charFunc(z - 1J, option.tau) / (z * (z - 1J))

def Andersen_Lake(model : HestonClass, option : VanillaOption) -> float:
    # calculating andersen lake parameters
    model.AndersenLakeParameters(option)
    alpha = optimal_alpha(model, option)
    
    def integrand(x):
        return (np.exp(-x * np.tan(model.phi) * model.omega + 1J * x * model.omega) * Q_H(-1J * alpha + x * (1 + 1J * np.tan(model.phi)), model, option) * (1 + 1J * np.tan(model.phi))).real
    
    integral = np.exp(alpha * model.omega) * quad(integrand, 0, np.inf)[0]
    
    R = model.forward * (alpha <= 0) - option.strike * (alpha <= -1) - 0.5 * (model.forward * (alpha == 0) - option.strike * (alpha == -1))

    return np.exp(-model.rate * option.tau) * (R - model.forward / np.pi * integral)

def calcImpVol(inputArray : np.array, optionList : np.array) -> np.ndarray:
    someModel = HestonClass(inputArray[0], inputArray[1], inputArray[2], inputArray[3], inputArray[4], inputArray[5], inputArray[6])
    outputLenght = np.shape(optionList)[0]
    output = np.empty(outputLenght, dtype=np.float64)
    for i in range(outputLenght):
        try:
            output[i] = someModel.impVol(Andersen_Lake(someModel, optionList[i]), optionList[i])
        except: #overflow in char function, set impvol to 0
            output[i] = 0

    return output

def impVolGenerator(inputArray : np.ndarray, optionList : np.array) -> np.ndarray:
    output_matrix = np.empty([np.shape(inputArray)[0], np.shape(optionList)[0]])
    i = 0
    for someInput in inputArray:
        output_matrix[i, :] = calcImpVol(someInput, optionList)
        i += 1
    
    return output_matrix

# Forward
forward = np.linspace(start = 75, stop = 125, num = 10)

# vol
vol = np.linspace(start = 0.01, stop = 0.1, num = 2)

# kappa
kappa = np.linspace(start = 0.1, stop = 2, num = 2)

# theta
theta = np.linspace(start = 0.01, stop = 0.1, num = 2)

# epsilon
epsilon = np.linspace(start = 0.1, stop = 2, num = 2)

# rho
rho = np.linspace(start = -0.99, stop = 0.99, num = 2)

# rate
rate = np.linspace(start = 0, stop = 0.2, num = 2)

# Maturity
maturity = np.linspace(start = 0.5, stop = 1, num = 2)

# strike
strike = np.linspace(start = 100, stop = 105, num = 2)

model_input = np.array(list(itertools.product(forward, vol, kappa, theta, epsilon, rho, rate))) # model parameter combinations
option_input = np.array(list(itertools.product(maturity, strike))) # different option combinations
someOptionList = np.array([])
for option in option_input:
    someOptionList = np.append(someOptionList, EUCall(option[0], option[1]))

# generating data for neural net with model as input and grid as output
input1 = model_input
start = time.time()
# going parallel
pool = Pool(64)
parallel_set = np.array_split(model_input, 64, axis=0)
parallel_input = [
    [parallel_set[0], someOptionList],
    [parallel_set[1], someOptionList],
    [parallel_set[2], someOptionList],
    [parallel_set[3], someOptionList],
    [parallel_set[4], someOptionList],
    [parallel_set[5], someOptionList],
    [parallel_set[6], someOptionList],
    [parallel_set[7], someOptionList],
    [parallel_set[8], someOptionList],
    [parallel_set[9], someOptionList],
    [parallel_set[10], someOptionList],
    [parallel_set[11], someOptionList],
    [parallel_set[12], someOptionList],
    [parallel_set[13], someOptionList],
    [parallel_set[14], someOptionList],
    [parallel_set[15], someOptionList],
    [parallel_set[16], someOptionList],
    [parallel_set[17], someOptionList],
    [parallel_set[18], someOptionList],
    [parallel_set[19], someOptionList],
    [parallel_set[20], someOptionList],
    [parallel_set[21], someOptionList],
    [parallel_set[22], someOptionList],
    [parallel_set[23], someOptionList],
    [parallel_set[24], someOptionList],
    [parallel_set[25], someOptionList],
    [parallel_set[26], someOptionList],
    [parallel_set[27], someOptionList],
    [parallel_set[28], someOptionList],
    [parallel_set[29], someOptionList],
    [parallel_set[30], someOptionList],
    [parallel_set[31], someOptionList],
    [parallel_set[32], someOptionList],
    [parallel_set[33], someOptionList],
    [parallel_set[34], someOptionList],
    [parallel_set[35], someOptionList],
    [parallel_set[36], someOptionList],
    [parallel_set[37], someOptionList],
    [parallel_set[38], someOptionList],
    [parallel_set[39], someOptionList],
    [parallel_set[40], someOptionList],
    [parallel_set[41], someOptionList],
    [parallel_set[42], someOptionList],
    [parallel_set[43], someOptionList],
    [parallel_set[44], someOptionList],
    [parallel_set[45], someOptionList],
    [parallel_set[46], someOptionList],
    [parallel_set[47], someOptionList],
    [parallel_set[48], someOptionList],
    [parallel_set[49], someOptionList],
    [parallel_set[50], someOptionList],
    [parallel_set[51], someOptionList],
    [parallel_set[52], someOptionList],
    [parallel_set[53], someOptionList],
    [parallel_set[54], someOptionList],
    [parallel_set[55], someOptionList],
    [parallel_set[56], someOptionList],
    [parallel_set[57], someOptionList],
    [parallel_set[58], someOptionList],
    [parallel_set[59], someOptionList],
    [parallel_set[60], someOptionList],
    [parallel_set[61], someOptionList],
    [parallel_set[62], someOptionList],
    [parallel_set[63], someOptionList]
]
res = pool.starmap(impVolGenerator, parallel_input)
output1 = np.concatenate(res, axis = 0)
stop = time.time()
print("time: ", stop-start)
np.savetxt("Data/testOutput.csv", output1, delimiter=",")