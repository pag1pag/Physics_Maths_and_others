"""
Estimation of a function, based on a polynom
The aim is to build this polynom by choosing correct coefficients
"""


import numpy as np
import matplotlib.pyplot as plt

# Sinus of x (with lot of points)
NP_POINTS1 = 100
X1 = np.linspace(0, 1, num=NP_POINTS1)
Y1 = np.sin(2 * np.pi * X1)

# Sinus of x (with less points)
N = int(NP_POINTS1 / 10)
X2 = np.linspace(0, 1, num=N)
Y2 = np.sin(2 * np.pi * X2)
# Sinus of x + some gaussian noise
T = Y2 + np.random.normal(0, 0.1, N)


POLYNOM_ORDER = 5


def y(x, w):
    """
    polynom of order m, which estimates the next value of f(x)
    knowing the precedent points (t1 = f(x1), t = f(x2), ...) (we don't know f)

    Attributes :
    ----
    x : int
        the value of which one wants the estimate
    w : array of size m
        weight of coefficents

    Returns:
    the sum of w_j * x**j,

    Example:
        >>> y(2, [0, 1, 2])
        >>> 'should be 0*2**0+1*2**1+2*2**2=0*2+1*2+2*4=0+2+8=10'
    """
    return np.sum([w[j] * (x ** j) for j in range(len(w))])


# The aim is to calculate W (vector of weights)
# It is found by minimizing the MSE

# matrix of size (n-1, m), because we don't know the last value
X = np.zeros((N - 1, POLYNOM_ORDER))
for i in range(N - 1):
    for j in range(POLYNOM_ORDER):
        X[i, j] = np.power(X2[i], j)

X_t = np.transpose(X)
A = X_t.dot(X)
A_inv = np.linalg.inv(A)
W = A_inv.dot(X_t).dot(T[:-1])


prediction = y(X2[-1], W)
true_value = T[-1]

# -- Some graphs -- #
plt.plot(X1, Y1, "r", label="true function")  # The function to estimate
# plt.plot(X2[:-1], Y2[:-1], 'or')
plt.plot(X2[:-1], T[:-1], "ob")  # The function + noise we have
plt.plot(X2[-1], prediction, "xr")
plt.plot(X2[-1], true_value, "xb")

# the function that we have estimate
tmp = np.zeros(NP_POINTS1)
for i in range(NP_POINTS1):
    tmp[i] = y(X1[i], W)
plt.plot(X1, tmp, "g", label="estimate function")

plt.legend()
plt.show()
