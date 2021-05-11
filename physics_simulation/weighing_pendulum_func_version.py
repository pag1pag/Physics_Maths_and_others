import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint
# plt.style.use('ggplot')

l = 1.0  # m
g = 9.81  # m/s**2


def f(X, t):
    """
    The derivative function
    """
    return np.array([X[1], -g / l * np.sin(X[0])])


def main():
    dt = 0.01
    X0 = np.radians(179)

    T = np.arange(0, 10, dt)
    theta0 = np.array([X0, -1])
    theta_odeint = odeint(f, theta0, T)[:, 0]

    theta_odeint = np.array(theta_odeint)
    X = l * np.sin(theta_odeint)
    Y = -l * np.cos(theta_odeint)

    fig, ax = plt.subplots()
    (rode,) = ax.plot([], [], color="blue")
    (point,) = ax.plot([], [], ls="none", marker="o", color="red")

    # Gestion des limites de la fenêtre
    # ax.axis("equal")
    ax.axis([1.05 * np.min(X), 1.05 * np.max(X), -1.05 * l, 1.05 * l])
    # ax.set_xlim([1.05 * np.min(X), 1.05 * np.max(X)])
    # ax.set_ylim([-1.05 * l, 1.05 * l])
    ax.set_aspect('equal', 'box')
    ax.vlines(x=0, ymin=-l, ymax=0, color="k", ls="--")
    angle = ax.text(0, -0.5, "",
                ha='center', va='center',
                fontsize=12)

    def init():
        rode.set_data([], [])
        point.set_data([], [])
        return rode, point

    def animate(k):
        i = min(k, X.size)
        x_rode = [len_rode * X[i] for len_rode in np.arange(0, l, l / 100)]
        y_rode = [len_rode * Y[i] for len_rode in np.arange(0, l, l / 100)]

        rode.set_data(x_rode, y_rode)
        point.set_data(X[i], Y[i])
        
        angle.set_text(f"{np.rad2deg(theta_odeint[i]):.1f}")

        return rode, point, angle

    #pylint: disable=unused-variable
    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=range(X.size),
        interval=50,
        blit=True,
    )

    plt.show()


main()
