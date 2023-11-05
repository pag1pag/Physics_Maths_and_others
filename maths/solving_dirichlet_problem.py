"""
Solving Direchlet problem on a square domain
============================================

We want to solve the following problem:

    Laplacian(u) = 0 in the square domain [0, 1]x[0, 1]
    u(x, y) = h(x, y) on the boundary of the square domain

We will use the grid transformation method to solve this problem.
"""

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from mpmath import ellipf, ellipfun, ellipk
from numba import njit
from scipy.integrate import quad

# Incomplete elliptic integral of the first kind
ellipf_vec = np.vectorize(ellipf, otypes=(complex,))
# TODO: see scipy.speciel.ellipkinc

# Jacobi elliptic function
cn = lambda u, k: ellipfun("cn", u, k=k)
cn = np.vectorize(cn, otypes=(complex,))
# TODO: see scipy.speciel.ellipj

# Complete elliptic integral of the first kind
ellipk_vec = np.vectorize(ellipk, otypes=(float,))
K_e = ellipk_vec(1 / 2)  # K_e = 1.854


# Constant for the grid transformation
C = 2 / K_e
A = 0
rot = np.exp(1j * np.pi / 4)
rot_inv = np.exp(-1j * np.pi / 4)


def inv_f(
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """An holomorphic function, transforming square to disk.
    The complex plan C is identified to R²: z = x + iy = (x, y)

    Args:
        x_coord (np.ndarray): real part.
        y_coord (np.ndarray): imaginary part.

    Returns:
        tuple[np.ndarray, np.ndarray]: image of (x, y) by f.
    """

    return rot_inv * cn(K_e - np.sqrt(2) / C * (rot * w - A), 1 / np.sqrt(2))


def f(
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """An holomorphic function, transforming disk to square.
    The complex plan C is identified to R²: z = x + iy = (x, y)

    Args:
        u_coord (np.ndarray): real part.
        v_coord (np.ndarray): imaginary part.

    Returns:
        tuple[np.ndarray, np.ndarray]: image of (u, v) by inv_f.
    """

    return rot_inv * (
        A + C / np.sqrt(2) * (K_e - ellipf_vec(np.arccos(rot * z), 1 / 2))
    )


@njit
def h(w) -> float:
    u = np.real(w)
    v = np.imag(w)

    if v > 0 and u > 0:
        return 1
    if v < 0:
        return -1
    return 0


@njit
def g(z) -> float:
    # x = np.real(z)
    y = np.imag(z)

    if y > 0:
        return 1
    if y < 0:
        return -1
    return 0


@njit
def integrand(phi, z):
    return (
        g(np.exp(1j * phi)) * (1 - np.abs(z) ** 2) / np.abs(z - np.exp(1j * phi)) ** 2
    )


def integrand2(phi, w):
    return (
        h(f(np.exp(1j * phi)))
        * (1 - np.abs(inv_f(w)) ** 2)
        / np.abs(inv_f(w) - np.exp(1j * phi)) ** 2
    )


def gamma(
    x_coord: np.ndarray,
    y_coord: np.ndarray,
) -> float:
    """Potential in the disk.

    Args:
        x_coord (np.ndarray): real part.
        y_coord (np.ndarray): imaginary part.

    Returns:
        tuple[np.ndarray, np.ndarray]: image of (x, y) by h.
    """
    z = x_coord + 1j * y_coord

    eps = 1e-10
    if np.abs(z) < 1 - eps:
        return quad(integrand, 0, 2 * np.pi, args=z)[0] / (2 * np.pi)
    return g(z)


def gamma2(
    u_coord: np.ndarray,
    v_coord: np.ndarray,
) -> float:
    w = u_coord + 1j * v_coord

    eps = 1e-10
    if np.abs(w) < 1 - eps:
        return quad(integrand2, 0, 2 * np.pi, args=w)[0] / (2 * np.pi)
    return h(w)


def T(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return 2 / np.pi * np.arctan((2 * r * np.sin(theta)) / (1 - r**2))


if __name__ == "__main__":
    # # Compute the potential on the boundary of the disk
    # r = np.linspace(0, 1, 100)
    # theta = np.linspace(0, 2 * np.pi, 100)
    # R, THETA = np.meshgrid(r, theta)

    # gamma_vec = np.vectorize(gamma, otypes=(float,))

    # Z = gamma_vec(R * np.cos(THETA), R * np.sin(THETA))
    # Z_true = T(R, THETA)

    # # Plot T on a circular domain (in 3D)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot_surface(R * np.cos(THETA), R * np.sin(THETA), Z)
    # ax.plot_surface(R * np.cos(THETA), R * np.sin(THETA), Z_true)
    # # plt.show()

    # # plot on 2d surface with countour plot and colorbar
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # tmp = ax.contourf(R * np.cos(THETA), R * np.sin(THETA), Z - Z_true)
    # plt.colorbar(tmp)

    # plt.tight_layout()
    # plt.axis("square")
    # plt.show()

    # Plot the potential on the boundary of the square
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, y)

    gamma2_vec = np.vectorize(gamma2, otypes=(float,))
    Z = gamma2_vec(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z)
    plt.show()
