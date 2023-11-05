"""2D Grid transformation on the grid [0, 1]x[0, 1]"""

from typing import Tuple

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from mpmath import ellipf, ellipfun
from scipy.special import ellipk

ellipf_vec = np.vectorize(ellipf, otypes=(complex,))

cn_m = lambda u, m: ellipfun("cn", u, m)
cn_m = np.vectorize(cn_m, otypes=(complex,))
z_test = 0.5 + 0.3j
print(cn_m(ellipf(np.arccos(z_test), 1 / np.sqrt(2)), 1 / np.sqrt(2)))
print(ellipf(np.arccos(cn_m(z_test, 1 / np.sqrt(2))), 1 / np.sqrt(2)))


cn = lambda u, k: ellipfun("cn", u, k=k)
cn = np.vectorize(cn, otypes=(complex,))

print(cn_m(z_test, 1 / 2), cn(z_test, 1 / np.sqrt(2)))

# Complete elliptic integral of the first kind
K_e = ellipk(1 / 2)  # K_e = 1.854
print(f"K_e = {K_e:.3f}")


def grid_transformation(
    x_coord: np.ndarray, y_coord: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """An holomorphic function: z -> 0.5*(z+0.5)**2
    The complex plan C is identified to R²: z = x + iy = (x, y)

    Args:
        x_coord (np.ndarray): real part.
        y_coord (np.ndarray): imaginary part.

    Returns:
        Tuple[np.ndarray, np.ndarray]: image of (x, y) by grid_transformation.
    """
    z = x_coord + 1j * y_coord

    w = np.sqrt(-1j) * cn(K_e * z * np.sqrt(1j / 2) - K_e, 1 / np.sqrt(2))

    # C = 2 / K_e
    # A = 0
    # rot = np.exp(1j * np.pi / 4)
    # rot_inv = np.exp(-1j * np.pi / 4)
    # w = rot_inv * cn(K_e - np.sqrt(2) / C * (rot * z - A), 1 / np.sqrt(2))

    return np.real(w), np.imag(w)


def inverse_grid_transformation(
    u_coord: np.ndarray, v_coord: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    w = u_coord + 1j * v_coord

    C = 2 / K_e
    A = 0
    rot = np.exp(1j * np.pi / 4)
    rot_inv = np.exp(-1j * np.pi / 4)
    z = rot_inv * (A + C / np.sqrt(2) * (K_e - ellipf_vec(np.arccos(rot * w), 1 / 2)))

    return np.real(z), np.imag(z)


def h_line(
    y_cst: float, nb_line_pts: int = 100, x_min=0, x_max=1
) -> Tuple[np.ndarray, np.ndarray]:
    """An horizontal line, described by y=y_cst, from x=0 to x=1.

    Args:
        y_cst (float): constant of the line equation.
        nb_line_pts: (int): number of coords to describe this line.

    Returns:
       Tuple[np.ndarray, np.ndarray]: the coords of this horizontal line.
    """
    return np.linspace(x_min, x_max, nb_line_pts), np.ones(nb_line_pts) * y_cst


def v_line(
    x_cst: float, nb_line_pts: int = 10, y_min=0, y_max=1
) -> Tuple[np.ndarray, np.ndarray]:
    """A vertical line, described by x=x_cst, from y=0 to y=1.

    Args:
        x_cst (float): constant of the line equation.
        nb_line_pts: (int): number of coords to describe this line.

    Returns:
       Tuple[np.ndarray, np.ndarray]: the coords of this vertical line.
    """
    return np.ones(nb_line_pts) * x_cst, np.linspace(y_min, y_max, nb_line_pts)


def plot_grid(
    v_steps: np.ndarray, h_steps: np.ndarray, x_min=0, x_max=1, y_min=0, y_max=1
) -> None:
    """Draw a grid.
    Each line are 1 unit long, and are spaced by nd.array.

    Args:
        steps (nd.array): position of lines.
    """
    for step in v_steps:
        # Draw original vertical lines
        plt.plot(*v_line(step, y_min=y_min, y_max=y_max), color="b", alpha=0.4, lw=0.5)

    for step in h_steps:
        # Draw original horizontal lines
        plt.plot(*h_line(step, x_min=x_min, x_max=x_max), color="b", alpha=0.4, lw=0.5)


def plot_transformed_grid(
    v_steps: np.ndarray, h_steps: np.ndarray, x_min=0, x_max=1, y_min=0, y_max=1
) -> None:
    """Draw the transformed grid.

    Args:
        steps (nd.array): position of lines.
    """
    for step in v_steps:
        # Draw transformed vertical lines
        plt.plot(
            *grid_transformation(*v_line(step, y_min=y_min, y_max=y_max)),
            color="r",
            alpha=0.8,
            lw=0.5,
        )

    for step in h_steps:
        # Draw transformed horizontal lines
        plt.plot(
            *grid_transformation(*h_line(step, x_min=x_min, x_max=x_max)),
            color="r",
            alpha=0.8,
            lw=0.5,
        )


def plot_inverse_transformed_grid(
    v_steps: np.ndarray, h_steps: np.ndarray, x_min=0, x_max=1, y_min=0, y_max=1
) -> None:
    for step in v_steps:
        # Draw transformed vertical lines
        plt.plot(
            *inverse_grid_transformation(
                *grid_transformation(*v_line(step, y_min=y_min, y_max=y_max))
            ),
            color="g",
            alpha=0.8,
            lw=0.5,
        )

    for step in h_steps:
        # Draw transformed horizontal lines
        plt.plot(
            *inverse_grid_transformation(
                *grid_transformation(*h_line(step, x_min=x_min, x_max=x_max))
            ),
            color="g",
            alpha=0.8,
            lw=0.5,
        )


def plot_top_line(nb_line_pts: int = 100) -> None:
    """Draw y=x*x on the normal grid, and after application by grid_transformation

    Args:
        nb_line_pts (int): number of coords to describe this line.
    """
    x_coords = np.linspace(1 / 3, 1, nb_line_pts)
    y_coords = np.ones(nb_line_pts)
    f_x, f_y = grid_transformation(x_coords, y_coords)
    plt.plot(x_coords, y_coords, color="b", alpha=0.5, label=r"$y=1$")
    plt.plot(f_x, f_y, color="r", alpha=1, label=r"$f_x, f_y = f(x, y)$")


if __name__ == "__main__":
    STEP = 40  # Each grid lines are 1/h away from each other

    x_min, x_max = -1, 1
    h_steps = np.linspace(-1, 1, STEP)

    y_min, y_max = -1, 1
    v_steps = np.linspace(-1, 1, STEP)

    # Plot one grid, and its transformation by grid_transformation
    plot_grid(v_steps, h_steps, x_min, x_max, y_min, y_max)
    plot_transformed_grid(v_steps, h_steps, x_min, x_max, y_min, y_max)

    # plot_inverse_transformed_grid(v_steps, h_steps, x_min, x_max, y_min, y_max)

    # Show the application of grid_transformation on the square function
    plot_top_line()

    # Add label, title, limits and legend on the plot
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        r"Grid transformation by the function $f(z)=e^{-\frac{j \pi}{4}} \text{cn}(K_e - \frac{\sqrt{2}}{C} (e^{\frac{j \pi}{4}} z - A), \frac{1}{\sqrt{2}})$"
    )

    plt.xlim(x_min - 0.1, x_max + 0.1)
    # square plot
    plt.axis("square")
    plt.ylim(y_min - 0.1, y_max + 0.1)

    plt.show()
