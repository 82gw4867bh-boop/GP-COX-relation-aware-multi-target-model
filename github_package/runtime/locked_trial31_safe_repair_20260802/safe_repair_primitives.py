import numpy as np
TAU = 0.25
def _s(x, tau=TAU):
    x = np.asarray(x, dtype=float)
    return np.sqrt(x * x + tau * tau) - tau
def soft_log(x, tau=TAU):
    return np.log1p(_s(x, tau))
def soft_sqrt(x, tau=TAU):
    return np.sqrt(_s(x, tau) + tau) - np.sqrt(tau)
def soft_div(a, b, tau=TAU):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a * b / (b * b + tau * tau)
