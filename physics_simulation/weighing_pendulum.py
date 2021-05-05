import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

l = 1.0  # m
g = 9.81  # m/s**2


def f(X, t):
    """
    The derivative function
    """
    return np.array([X[1], -g / l * np.sin(X[0])])


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


def main():
    T = np.arange(0, 10, 0.01)
    X0 = np.array([np.radians(30), 0])
    print(X0)
    X_euler = Euler(f, X0, T)[:, 0]
    X_RK4 = RK4(f, X0, T)[:, 0]
    X_odeint = odeint(f, X0, T)[:, 0]

    plt.plot(T, X_euler, label="Euler")
    plt.plot(T, X_RK4, "x", label="RK4")
    plt.plot(T, X_odeint, label="odeint")
    plt.legend()
    plt.show()


main()
