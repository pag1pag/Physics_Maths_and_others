"""
Solving Direchlet problem on a square domain
============================================

We want to solve the following problem:

    Laplacian(u) = 0 in the square domain [-1, 1]x[-1, 1]
    u(x, y) = h(x, y) on the boundary of the square domain

We will use the grid transformation method to solve this problem.

Points in the disk (D) are denoted by z = x + iy,
points in the square (S) are denoted by w = u + iv.

The function f is an holomorphic function, transforming the disk (D) into the square (S).
The function inv_f is the inverse of f (S -> D).

Note:
    We could have use Serie expansion to solve this problem.
    It is analytically solvable on a rectangle.

# TODO: optimize the code since it is very slow :/ (or use C++ and cython)

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
) -> np.ndarray:
    r"""An holomorphic function, transforming a square (S) into a disk (D).

    .. math::
        z = f^{-1}(w) = e^{-\frac{i \pi}{4}} \text{cn}(K_e - \frac{\sqrt{2}}{C} (e^{\frac{i \pi}{4}} w - A), \frac{1}{\sqrt{2}})

    Note:
        cn is a Jacobi elliptic function.

    Args:
        w (np.ndarray): Complex number, representing a point in the square.

    Returns:
        np.ndarray: Complex number, representing a point in the disk.
    """

    return rot_inv * cn(K_e - np.sqrt(2) / C * (rot * w - A), 1 / np.sqrt(2))


def f(
    z: np.ndarray,
) -> np.ndarray:
    r"""An holomorphic function, transforming a disk (D) into square (S).

    .. math::
        e^{-\frac{i \pi}{4}} (A + \frac{C}{\sqrt{2}}(K_e - \text{F}(\arccos(e^{\frac{i \pi}{4}} z) | \frac{1}{2})))

    Note:
        F is the incomplete elliptic integral of the first kind.

    Args:
        z (np.ndarray): Complex number, representing a point in the disk.

    Returns:
        np.ndarray: Complex number, representing a point in the square.
    """

    return rot_inv * (
        A + C / np.sqrt(2) * (K_e - ellipf_vec(np.arccos(rot * z), 1 / 2))
    )


@njit
def h(w: np.ndarray) -> float:
    """Boundary condition on the square.

    Args:
        w (np.ndarray): Complex number, representing a point in the square.

    Returns:
        float: Boundary condition on the square.
    """
    u, v = np.real(w), np.imag(w)

    if np.isclose(v, 1):  # Top
        if u < 0:
            return 0
        if 0 < u < 2 / 3:
            return 75 * u
        if 2 / 3 < u < 1:
            return 150 * (1 - u)
    if np.isclose(v, -1):  # Bottom
        return 0
    if np.isclose(u, 1):  # Right
        return 0
    if np.isclose(u, -1):  # Left
        return 0


@njit
def g_test(z: np.ndarray) -> float:
    """Boundary condition on the disk (used for testing).

    Args:
        z (np.ndarray): Complex number, representing a point in the disk.

    Returns:
        float: Boundary condition on the disk.
    """

    # x = np.real(z)
    y = np.imag(z)

    if y > 0:
        return 1
    if y < 0:
        return -1
    return 0


@njit
def integrand_disk(psi: float, z: complex) -> float:
    """Integrand for the potential on the boundary of the disk.

    See :func:`gamma_disk` for more details.
    """

    return (
        g_test(np.exp(1j * psi))
        * (1 - np.abs(z) ** 2)
        / np.abs(z - np.exp(1j * psi)) ** 2
    )


def integrand_square(psi: float, w: complex) -> float:
    r"""Integrand for the potential on the boundary of the square.

    This function defined an integrand:

    .. math::
        h(f(e^{i \psi})) \frac{1 - \abs{f^{-1}(w)}^2}{\abs{f^{-1}(w)-e^{i \psi}}^2}

    See :func:`gamma_square` for more details.
    """

    return (
        h(f(np.exp(1j * psi)))
        * (1 - np.abs(inv_f(w)) ** 2)
        / np.abs(inv_f(w) - np.exp(1j * psi)) ** 2
    )


def gamma_disk(
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
        return quad(integrand_disk, 0, 2 * np.pi, args=z)[0] / (2 * np.pi)
    return g_test(z)


def gamma_square(
    u_coord: np.ndarray,
    v_coord: np.ndarray,
) -> float:
    r"""Solve the Dirichlet problem on the square.

    On the disk, the solution is given by the following formula:

    .. math::
        \Theta(z) = \frac{1}{2 \pi} \int_0^{2\pi} g(e^{i \psi}) \frac{1 - \abs{z}^2}{\abs{z-e^{i \psi}}^2} d\psi

    On the square, the solution is given by the following formula:

    .. math::
        \Gamma(w) = \frac{1}{2 \pi} \int_0^{2\pi} h(f(e^{i \psi})) \frac{1 - \abs{f^{-1}(w)}^2}{\abs{f^{-1}(w)-e^{i \psi}}^2} d\psi

    Args:
        u_coord (np.ndarray): x coordinate (or real part).
        v_coord (np.ndarray): y coordinate (or imaginary part).

    Returns:
        float: Potential on the square at the point (u, v).
    """

    w = u_coord + 1j * v_coord

    eps = 1e-10
    if np.abs(w) < 1 - eps:
        return quad(integrand_square, 0, 2 * np.pi, args=w)[0] / (2 * np.pi)
    return h(w)


def T(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """True solution of the Dirichlet problem on the disk, with boundary condition g_test.

    Args:
        r (np.ndarray): radius.
        theta (np.ndarray): angle.

    Returns:
        np.ndarray: True solution of the Dirichlet problem on the disk.
    """

    return 2 / np.pi * np.arctan((2 * r * np.sin(theta)) / (1 - r**2))


if __name__ == "__main__":
    # # Compute the potential on the boundary of the disk
    # r = np.linspace(0, 1, 100)
    # theta = np.linspace(0, 2 * np.pi, 100)
    # R, THETA = np.meshgrid(r, theta)

    # gamma_vec = np.vectorize(gamma_disk, otypes=(float,))

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

    gamma2_vec = np.vectorize(gamma_square, otypes=(float,))
    Z = gamma2_vec(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z)
    plt.show()
