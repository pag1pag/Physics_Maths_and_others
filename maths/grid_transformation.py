"""2D Grid transformation on the grid [0, 1]x[0, 1]"""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt  # type: ignore


def grid_transformation(x_coord: np.ndarray, y_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """An holomorphic function: z -> 0.5*(z+0.5)**2
    The complex plan C is identified to R²: z = x + iy = (x, y)

    Args:
        x_coord (np.ndarray): real part.
        y_coord (np.ndarray): imaginary part.

    Returns:
        Tuple[np.ndarray, np.ndarray]: image of (x, y) by grid_transformation.
    """
    z_complex = (x_coord + 1j * y_coord + 1 / 2) ** 2 / 2
    return np.real(z_complex), np.imag(z_complex)


def h_line(y_cst: float, nb_line_pts: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """An horizontal line, described by y=y_cst, from x=0 to x=1.

    Args:
        y_cst (float): constant of the line equation.
        nb_line_pts: (int): number of coords to describe this line.

    Returns:
       Tuple[np.ndarray, np.ndarray]: the coords of this horizontal line.
    """
    return np.linspace(0, 1, nb_line_pts), np.ones(nb_line_pts) * y_cst


def v_line(x_cst: float, nb_line_pts: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """A vertical line, described by x=x_cst, from y=0 to y=1.

    Args:
        x_cst (float): constant of the line equation.
        nb_line_pts: (int): number of coords to describe this line.

    Returns:
       Tuple[np.ndarray, np.ndarray]: the coords of this vertical line.
    """
    return np.ones(nb_line_pts) * x_cst, np.linspace(0, 1, nb_line_pts)


def plot_grid(steps: np.ndarray) -> None:
    """Draw a grid.
    Each line are 1 unit long, and are spaced by nd.array.

    Args:
        steps (nd.array): position of lines.
    """
    for step in steps:
        # Draw original vertical lines
        plt.plot(*v_line(step), color="b", alpha=0.4, lw=0.5)
        # Draw original horizontal lines
        plt.plot(*h_line(step), color="b", alpha=0.4, lw=0.5)


def plot_transformed_grid(steps: np.ndarray) -> None:
    """Draw the transformed grid.

    Args:
        steps (nd.array): position of lines.
    """
    for step in steps:
        # Draw transformed vertical lines
        plt.plot(*grid_transformation(*v_line(step)),
                 color="r", alpha=0.8, lw=0.5)
        # Draw transformed horizontal lines
        plt.plot(*grid_transformation(*h_line(step)),
                 color="r", alpha=0.8, lw=0.5)


def plot_square_function(nb_line_pts: int = 100) -> None:
    """Draw y=x*x on the normal grid, and after application by grid_transformation

    Args:
        nb_line_pts (int): number of coords to describe this line.
    """
    x_coords = np.linspace(0, 1, nb_line_pts)
    y_coords = x_coords * x_coords
    f_x, f_y = grid_transformation(x_coords, y_coords)
    plt.plot(x_coords, y_coords, color="b", alpha=0.5, label=r"$y=x^2$")
    plt.plot(f_x, f_y, color="r", alpha=1, label=r"$f_x, f_y = f(x, y)$")


if __name__ == "__main__":
    STEP = 10  # Each grid lines are 1/h away from each other
    line_steps = np.linspace(0, 1, STEP)

    # Plot one grid, and its transformation by grid_transformation
    plot_grid(line_steps)
    plot_transformed_grid(line_steps)

    # Show the application of grid_transformation on the square function
    plot_square_function()

    # Add label, title, limits and legend on the plot
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        r"Grid transformation by the function $f(z)=\frac{1}{2}(z+\frac{1}{2})^2$"
    )
    plt.xlim(-0.4, 1.2)
    plt.ylim(-0.1, 1.6)
    plt.legend()

    plt.show()
