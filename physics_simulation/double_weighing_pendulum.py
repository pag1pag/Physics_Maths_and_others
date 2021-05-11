from random import choice

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint


def f(X, t):
    """
    The derivative function
    """
    # theta1 = X[0]
    # theta2 = X[1]
    # theta1dot = X[2]
    # theta2dot = X[3]

    a = -m2 * l2 * X[3] * X[3] * np.sin(X[0] - X[1])
    b = -(m1 + m2) * g * np.sin(X[0])
    c = -m2 * l1 * X[2] * X[2] * np.sin(X[0] - X[1]) * np.cos(X[0] - X[1])
    d = m2 * g * np.sin(X[1]) * np.cos(X[0] - X[1])
    num = a + b + c + d
    den = (m1 + m2) * l1 - m2 * l1 * np.cos(X[0] - X[1]) * np.cos(X[0] - X[1])
    theta1dotdot = num / den

    a = a * np.cos(X[0] - X[1])
    b = b * np.cos(X[0] - X[1])
    c = -(m1 + m2) * l1 * X[2] * X[2] * np.sin(X[0] - X[1])
    d = (m1 + m2) * g * np.sin(X[1])
    num = a + b + c + d
    den = m2 * l2 * np.cos(X[0] - X[1]) * np.cos(X[0] - X[1]) - (m1 + m2) * l2
    theta2dotdot = num / den

    return np.array([X[2], X[3], theta1dotdot, theta2dotdot])


def main(with_drag: bool=True, *X0s: list) -> None:
    """
    Solve for initial conditions X0s and draw an animation of the double pendulum.
    There could as many IC has one wants (but a least one), and their evolution is drawn on the same graph

    Args:
        - with_drag (bool): display (or not) a drag behind the second point/mass
        - *X0s (n-uple of list): Each list correspond to one inital conditions (initial_theta1, initial_theta2, initial_theta1dot, initial_theta2dot)

    Returns:
        None, just run an animation
    """
    dt = 0.01
    T = np.arange(0, 100, dt)

    def get_coords(X0):
        """
        solve the equation for the current X0

        Args:
            - X0 (list) : initial conditions. Here, list of size 4

        Returns:
        4 lists of coords, with their evolution over time
            - X1 : distance to center on x-axis
            - Y1 : distance to center on y-axis
            - X2 : distance to X1 on x-axis
            - Y2 : distance to Y1 on y-axis
        """
        solution = odeint(f, X0, T)

        theta1 = np.array(solution[:, 0])
        theta2 = np.array(solution[:, 1])

        X1 = l1 * np.sin(theta1)
        Y1 = -l1 * np.cos(theta1)
        X2 = l2 * np.sin(theta2)
        Y2 = -l2 * np.cos(theta2)

        return X1, Y1, X2, Y2

    def get_rodes_points():
        """
        Create graphical object for better visualisation.
        rodes are blue, points are red (sugar is sweet, and so are you)
        """
        (rode1,) = ax.plot([], [], color="blue")
        (point1,) = ax.plot([], [], ls="none", marker="o", color="red")
        (rode2,) = ax.plot([], [], color="blue")
        (point2,) = ax.plot([], [], ls="none", marker="o", color="red")
        return rode1, point1, rode2, point2

    def get_drag():
        """maybe not the best word"""
        shades_of_green = ["yellowgreen", "greenyellow", "chartreuse", "lightgreen", "forestgreen", "green", "lime", "seagreen"]
        (drag,) = ax.plot([], [], color=choice(shades_of_green))
        return drag

    def init_rodes_points(*rodes_or_points):
        """
        for each element of rodes_or_points, reset data

        Args:
            rodes_or_points (list of rode or point (defined by get_rodes_points))
        """
        for rode_or_point in rodes_or_points:
            rode_or_point.set_data([], [])

    def animate_rodes_points(i, X1, Y1, X2, Y2, rode1, point1, rode2, point2):
        """
        This function aims to change the position of every elements (each points, each rodes)

        Args:
            - i (int): instant/frame to display
            - X1, Y1, X2, Y2 (lists): coords of each points
            - rode1, point1, rode2, point2 (...): elements to set data to, and that are displayed
        """
        x_rode1 = [len_rode * X1[i] for len_rode in np.arange(0, l1, l1 / 100)]
        y_rode1 = [len_rode * Y1[i] for len_rode in np.arange(0, l1, l1 / 100)]

        x_rode2 = [
            l1 * X1[i] + len_rode * X2[i] for len_rode in np.arange(0.03, l2, l2 / 100)
        ]
        y_rode2 = [
            l1 * Y1[i] + len_rode * Y2[i] for len_rode in np.arange(0.03, l2, l2 / 100)
        ]

        rode1.set_data(x_rode1, y_rode1)
        point1.set_data(X1[i], Y1[i])

        rode1.set_data(x_rode1, y_rode1)
        point1.set_data(X1[i], Y1[i])

        rode2.set_data(x_rode2, y_rode2)
        point2.set_data(X1[i] + X2[i], Y1[i] + Y2[i])

    def animate_drags(i, X1, Y1, X2, Y2, *drags):
        n = 100
        for drag in drags:
            if i > n:
                drag.set_data(X1[i-n:i] + X2[i-n:i], Y1[i-n:i] + Y2[i-n:i])
            else:
                drag.set_data(X1[:i] + X2[:i], Y1[:i] + Y2[:i])

    def init():
        """
        for each double-pendulum, called the init function
        """
        all_graphical_elements = []

        if with_drag:
            for drags in all_drags:
                init_rodes_points(drags)
                all_graphical_elements.append(drags)


        for rodes_points in all_rodes_points:
            init_rodes_points(*rodes_points)
            for rode_point in rodes_points:
                all_graphical_elements.append(rode_point)



        return tuple(all_graphical_elements)

    def animate(k):
        """
        for each double-pendulum, called the animate function
        """
        all_graphical_elements = []

        if with_drag:
            for coords, drags in zip(all_coords, all_drags):
                animate_drags(k, *coords, drags)
                all_graphical_elements.append(drags)

        for coords, rodes_points in zip(all_coords, all_rodes_points):
            animate_rodes_points(k, *coords, *rodes_points)
            for rode_point in rodes_points:
                all_graphical_elements.append(rode_point)



        return tuple(all_graphical_elements)

    if X0s is None:
        raise ValueError("X0s should not be None")

    # set data
    all_coords = []
    all_rodes_points = []
    all_drags = []
    for X0 in X0s:
        x1, y1, x2, y2 = get_coords(X0)
        all_coords.append((x1, y1, x2, y2))

        rode1, point1, rode2, point2 = get_rodes_points()
        all_rodes_points.append((rode1, point1, rode2, point2))

        if with_drag:
            drag = get_drag()
            all_drags.append(drag)

    # create the animation
    # pylint: disable=unused-variable
    anim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        frames=range(T.size),
        init_func=init,
        interval=5,
        blit=True,
    )

    # save 
    # anim.save("pendule_double.mp4")

    # run the animation
    plt.show()


if __name__ == "__main__":
    # CONSTANTS
    l1 = 1.0  # m
    m1 = 1.0  # kg
    l2 = 1.0  # m
    m2 = 1.0  # kg
    g = 9.81  # m/s**2

    # Graph
    fig, ax = plt.subplots()
    (x_min, x_max) = (-1.05 * (l1 + l2), 1.05 * (l1 + l2))
    (y_min, y_max) = (-1.05 * (l1 + l2), 1.05 * (l1 + l2))
    ax.axis([x_min, x_max, y_min, y_max])
    ax.set_aspect("equal", "box")

    # Various initial conditions
    X0_1 = [np.radians(90), np.radians(89), 0, 0]
    X0_2 = [np.radians(90), np.radians(90), 0, 0]
    X0_3 = [np.radians(90), np.radians(91), 0, 0]

    main(True, X0_1, X0_2, X0_3)
