"""Simulation of the heating up of a copper rod

Initial and boundary conditions are:
- IC: the rod is at ambiant temperature Ta at instant 0,
- left BC: the left side of the rod is instantaneously heat up at temperature Tc for a period of tc seconds,
- right BC: the right side of the rod is kept isolated.

Source :
https://perso.univ-rennes1.fr/fabrice.mahe/ens/applications/chaleur1d/chaleur1d.html
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import animation


def left_boundary_condition(time):
    """
    Define the left boundary condition of the rod.
    Here is a modelisation of an instantaneous heating up (for a period of tc seconds).
    """
    if time < tc:
        return Tc
    return Ta


def get_differential_matrix():
    """
    Matrix of the thermal diffusion, taking into account right bondary conditions.
    A retrograde Euler scheme is used here.
    """
    diff_matrix = np.diag(np.full(N, 2.0), k=0)+np.diag(np.full(N-1, -1.0),
                                                        k=1)+np.diag(np.full(N-1, -1.0), k=-1)
    diff_matrix[N - 1, N - 2] = -2  # right bondary conditions.
    return diff_matrix


def get_temperature():
    """
    return the temperature of the bar at each instant
    temperature is a N*M matrix, where:
    - the first column is the initial state of the rod
    - the second one is the second state of the rod (rod at time tau)
    - the third column represents the rod at time 2*tau
    - and so on...
    """

    A = get_differential_matrix()
    I = np.identity(N)
    Inv = linalg.inv(I + k * A)

    temperature = np.zeros((N, M))
    # Initial condition
    temperature[:, 0] = Ta

    # matrix of previous state
    F = np.zeros((N, M))
    F[:, 0] = Ta
    F[0, 0] = Ta + k * left_boundary_condition(tau)

    # solving thermal diffusion
    for j in range(1, M):
        temperature[:, j] = Inv.dot(F[:, j - 1])
        F[:, j] = temperature[:, j]
        F[0, j] = temperature[0, j] + k * \
            left_boundary_condition(j * (tau + 1))

    return temperature


def run_animation():
    """Launch the animation of the heating of the rod"""

    # creation of the rod: each points on the same x-axis have the same temperature
    rod = np.zeros((N, N))
    for i in range(N):
        rod[i] = T[:, 0]
    # print(rod)
    # print(rod.shape)

    # creation of the figure
    fig = plt.figure()
    fig.suptitle("Evolution of temperature in rode over time")
    # creation of the axus
    axis = plt.axes(xlim=(0, L), ylim=(0, L / 10))
    axis.axes.get_yaxis().set_visible(False)
    axis.set_xlabel("distance (m)")
    # creation of the image
    image = plt.imshow(
        rod,
        interpolation="bicubic",
        origin="lower",
        extent=[0, L, 0, L / 10],
        cmap="RdBu_r",
        vmin=Ta,
        vmax=Tc,
    )
    # creation of the colorbar
    cbar = plt.colorbar(extend="both")
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("Temperature (K)", rotation=270)

    def init():
        """plot the background of each frame"""
        image.set_data(np.zeros((N, N)))
        return [image]

    def animate(frame_j):
        """This is called sequentially"""
        print(frame_j)
        t = frame_j * tau
        rod = image.get_array()
        for i in range(N):
            rod[i] = T[:, t]
        image.set_array(rod)
        return [image]

    # creation of the animation
    # pylint: disable=unused-variable
    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=M, blit=True, interval=10, repeat=False
    )

    plt.show()


if __name__ == "__main__":
    # time discretisation
    tau = 1  # s (time step between two snapshots)
    M = 1000  # number of points
    tc = 600  # s (duration of heating)

    # space discretisation
    L = 0.5  # cm (length of the rod)
    N = 4  # how many subdivision of the rod
    h = L / N  # cm (footstep)

    # D: thermal coefficient
    # D = alpha * rho * cp
    D = 1.15e-4  # m²/s (copper)
    # k: dimensionless thermal coefficient
    k = D * tau / (h * h)

    # initial temperature
    Ta = 273 + 20  # K
    # heating temperature (in the left )
    Tc = 273 + 100  # K

    T = get_temperature()

    run_animation()
