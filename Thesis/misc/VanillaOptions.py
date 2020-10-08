import numpy as np

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