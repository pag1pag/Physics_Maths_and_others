import numpy as np

def Euler(func, X0, t):
    """
    Euler integrator.
    """
    dt = t[1] - t[0]
    nt = len(t)
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt - 1):
        X[i + 1] = X[i] + func(X[i], t[i]) * dt
    return X


def RK4(func, X0, t):
    """
    Runge and Kutta 4 integrator.
    """
    dt = t[1] - t[0]
    nt = len(t)
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt - 1):
        k1 = func(X[i], t[i])
        k2 = func(X[i] + dt / 2.0 * k1, t[i] + dt / 2.0)
        k3 = func(X[i] + dt / 2.0 * k2, t[i] + dt / 2.0)
        k4 = func(X[i] + dt * k3, t[i] + dt)
        X[i + 1] = X[i] + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return X