import numpy as np

# Generating BlackScholes normalized dataset for fixed strike, sigma and maturity
def BSDataGenerator(strike : int, sigma : float, mat : float, length : int): 
    X = strike * np.exp(-sigma/2 * mat + sigma * np.sqrt(mat) * np.random.normal(0, 1, length))

    ST = X * np.exp(-sigma * sigma/2 * mat + sigma * np.sqrt(mat) * np.random.normal(0, 1, length))

    Y = np.maximum(ST - strike, 0)

    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    return X, Y