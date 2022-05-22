import numpy as np


def euler(func, initial_conditions, t):
    """
    Euler integrator.
    """
    dt = t[1] - t[0]
    nt = len(t)
    x = np.zeros([nt, len(initial_conditions)])
    x[0] = initial_conditions
    for i in range(nt - 1):
        x[i + 1] = x[i] + func(x[i], t[i]) * dt
    return x


def runge_kutta_4(func, initial_conditions, t):
    """
    Runge and Kutta 4 integrator.
    """
    dt = t[1] - t[0]
    nt = len(t)
    x = np.zeros([nt, len(initial_conditions)])
    x[0] = initial_conditions
    for i in range(nt - 1):
        k1 = func(x[i], t[i])
        k2 = func(x[i] + dt / 2.0 * k1, t[i] + dt / 2.0)
        k3 = func(x[i] + dt / 2.0 * k2, t[i] + dt / 2.0)
        k4 = func(x[i] + dt * k3, t[i] + dt)
        x[i + 1] = x[i] + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x
