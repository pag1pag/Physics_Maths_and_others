from typing import List, Tuple
from random import choice

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import odeint


class DoublePendulum:
    """This class implements the physics behind the double pendulum"""

    def __init__(self, initial_conditions: List[float],
                 l1: float = 1.0, l2: float = 1.0,
                 m1: float = 1.0, m2: float = 1.0,
                 g: float = 9.81,
                 t_max: float = 100, dt: float = 0.01) -> None:
        """Initialisation function

        Args:
            initial_conditions (List[float]): A list of four elements.
                The 1st is the initial angle between the first rode and the -y-axis [rad].
                The 2nd is the initial angle between the second rode and the -y-axis [rad].
                The 3rd is the initial angular velocity of the first rode [rad.s^-1].
                The 4th is the initial angular velocity of the second rode [rad.s^-1].
            l1 (float, optional): length of the first rode [m]. Defaults to 1.0.
            l2 (float, optional): length of the second rode [m]. Defaults to 1.0.
            m1 (float, optional): mass of the first rode [kg]. Defaults to 1.0.
            m2 (float, optional): mass of the second rode [kg]. Defaults to 1.0.
            g (float, optional): gravitational constant [m.s^-2]. Defaults to 9.81.
            t_max (float, optional): duration of the simulation [s]. Defaults to 100.
            dt (float, optional): timestep of simulation [s]. Defaults to 0.01.

        Raises:
            ValueError: initial_conditions should be a list of four elements
        """

        if not isinstance(initial_conditions, list):
            raise ValueError("initial_conditions should be a list")
        if len(initial_conditions) != 4:
            raise ValueError("Should have 4 elements in initial_conditions")
        # Initial conditions
        self.initial_conditions = initial_conditions

        # Physical parameters
        self.l1 = l1  # length of the first rode
        self.l2 = l2  # length of the second rode
        self.m1 = m1  # length of the first rode
        self.m2 = m2  # mass of the first point
        self.g = g    # mass of the second point

        # Time paramaters
        self.times = np.arange(0, t_max, dt)  # Duration

        # coords of the points
        self.x1 = np.zeros(self.times.shape)
        self.y1 = np.zeros(self.times.shape)
        self.x2 = np.zeros(self.times.shape)
        self.y2 = np.zeros(self.times.shape)

        self.solve_and_set_coords()

    def f(self, x: np.ndarray, _: float) -> np.ndarray:
        """Ordinary differential equation: Xdot = f(X, t), with
        - X[0] = theta1
        - X[1] = theta2
        - X[2] = theta1dot
        - X[3] = theta2dot

        - Xdot[0] = theta1dot = X[2]
        - Xdot[1] = theta2dot = X[3]
        - Xdot[2] = theta1dotdot = ...
        - Xdot[3] = theta2dotdot = ...

        To find the expression of theta1dotdot and theta2dotdot, a Lagrangian
        approach was taken.

        Args:
            X (np.ndarray): Input vector
            _ (float): time (but not use here)

        Returns:
            np.array: Xdot, whose terms are linked to X by f
        """

        a = -self.m2 * self.l2 * x[3] * x[3] * np.sin(x[0] - x[1])
        b = -(self.m1 + self.m2) * self.g * np.sin(x[0])
        c = -self.m2 * self.l1 * x[2] * x[2] * \
            np.sin(x[0] - x[1]) * np.cos(x[0] - x[1])
        d = self.m2 * self.g * np.sin(x[1]) * np.cos(x[0] - x[1])
        num = a + b + c + d
        den = (self.m1 + self.m2) * self.l1 - self.m2 * \
            self.l1 * np.cos(x[0] - x[1]) * np.cos(x[0] - x[1])
        theta1dotdot = num / den

        a = a * np.cos(x[0] - x[1])
        b = b * np.cos(x[0] - x[1])
        c = -(self.m1 + self.m2) * self.l1 * x[2] * x[2] * np.sin(x[0] - x[1])
        d = (self.m1 + self.m2) * self.g * np.sin(x[1])
        num = a + b + c + d
        den = self.m2 * self.l2 * \
            np.cos(x[0] - x[1]) * np.cos(x[0] - x[1]) - \
            (self.m1 + self.m2) * self.l2
        theta2dotdot = num / den

        return np.array([x[2], x[3], theta1dotdot, theta2dotdot])

    def solve_and_set_coords(self) -> None:
        """Solve f using scipy.integrate.odeint, with initial conditions initial_conditions, over the period 'time'
        """
        solution = odeint(self.f, self.initial_conditions, self.times)

        # angle between the -y axis and the first pendulum
        theta1 = np.array(solution[:, 0])
        # angle between the -y axis and the second pendulum
        theta2 = np.array(solution[:, 1])

        self.x1 = self.l1 * np.sin(theta1)
        self.y1 = -self.l1 * np.cos(theta1)
        self.x2 = self.l2 * np.sin(theta2)
        self.y2 = -self.l2 * np.cos(theta2)

    def get_pos_rodes_and_points(self, t: int) -> Tuple[
            Tuple[np.ndarray, np.ndarray],
            Tuple[float, float],
            Tuple[np.ndarray, np.ndarray],
            Tuple[float, float]]:
        """
        Calculate the position of the two points and the two rodes at time t.

        Args:
            - t (int): instant/frame to display

        Returns:
            The (x, y) coordinates of the two points, and the list of (x, y) coordinates describing the rodes, at time t
        """
        x_rode1 = np.array([len_rode * self.x1[t]/self.l1
                            for len_rode in np.arange(0, self.l1, self.l1 / 100)])
        y_rode1 = np.array([len_rode * self.y1[t]/self.l1
                            for len_rode in np.arange(0, self.l1, self.l1 / 100)])

        x_rode2 = np.array([
            self.x1[t] + len_rode * self.x2[t]/self.l2 for len_rode in np.arange(0.03, self.l2, self.l2 / 100)
        ])
        y_rode2 = np.array([
            self.y1[t] + len_rode * self.y2[t]/self.l2 for len_rode in np.arange(0.03, self.l2, self.l2 / 100)
        ])

        rode1 = (x_rode1, y_rode1)
        point1 = (self.x1[t], self.y1[t])
        rode2 = (x_rode2, y_rode2)
        point2 = (self.x1[t] + self.x2[t], self.y1[t] + self.y2[t])

        return (rode1, point1, rode2, point2)

    def get_drag(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the drag behind the pendulum

        Args:
            t (int): instant/frame to display

        Returns:
            Tuple[np.ndarray, np.ndarray]: the (x, y) coords of past time
        """
        # delay/nb of previous points still on screen
        n = 100
        if t > n:
            return self.x1[t-n:t] + self.x2[t-n:t], self.y1[t-n:t] + self.y2[t-n:t]
        return self.x1[:t] + self.x2[:t], self.y1[:t] + self.y2[:t]


class AnimateDoublePendulum:
    """A class to animate multiple double pendulum on the same graph."""

    def __init__(self, double_pendulums: List[DoublePendulum], with_drag: bool = True, save: bool = False) -> None:
        """Init function

        Args:
            double_pendulums (List[DoublePendulum]): A list of double pendulums to animate on the same graph
            with_drag (bool, optional): display a drag behind the double pendulum. Defaults to True.
            save (bool, optional): save the animation. Defaults to False.

        Raises:
            ValueError: double_pendulum should be a list of DoublePendulum elements
        """

        if not isinstance(double_pendulums, list):
            raise ValueError("double_pendulums should be a list")
        for dp in double_pendulums:
            if not isinstance(dp, DoublePendulum):
                raise ValueError(
                    "double_pendulums should be a list of DoublePendulum")
        self.double_pendulums = double_pendulums

        self.with_drag = with_drag

        # The figure on which to display the animation
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.set_graph()

        # -- The graphical elements to display on the figure --#
        # This corresponds to the points (in red), and the lines (in red)
        self.all_rodes_points: List[Tuple[Line2D, ...]] = []
        self.create_rodes_and_points()

        # This corresponds to the drags (in green)
        self.all_drags: List[Line2D] = []
        if self.with_drag:
            self.create_drags()

        self.run_animation(save=save)

    def run_animation(self, save: bool = False) -> None:
        """The main function of this class"""
        # run the simulation only over the smallest period of time
        # (if they are double pendulum with different simulation time, choose the smallest of this time)
        min_simulation_time = np.min(
            [dp.times.size for dp in self.double_pendulums])

        anim = animation.FuncAnimation(
            fig=self.fig,
            func=self.animate,
            frames=range(min_simulation_time),
            init_func=self.init,
            interval=5,  # Delay between frames in milliseconds.
            blit=True,
        )

        if save:
            anim.save("double_pendulum.mp4")
        else:
            # run the animation
            plt.show()

    def create_rodes_and_points(self) -> None:
        """For each double pendulum, create 4 elements (2 rodes and 2 points).
        Rodes are blue, points are red (sugar is sweet, and so are you).

        Here, we only create the elements, but they don't have data inside.
        """
        for _ in self.double_pendulums:
            (rode1,) = self.ax.plot([], [], color="blue")
            (point1,) = self.ax.plot([], [], ls="none", marker="o", color="red")
            (rode2,) = self.ax.plot([], [], color="blue")
            (point2,) = self.ax.plot([], [], ls="none", marker="o", color="red")

            # self.all_rodes_points is a list of tuple
            self.all_rodes_points.append((rode1, point1, rode2, point2))

    def create_drags(self) -> None:
        """For each double pendulum, create a drag behind it (if enabled).
        Drags are green.

        Here, we only create the drags, but they don't have data inside.
        """
        shades_of_green = ["yellowgreen", "greenyellow", "chartreuse",
                           "lightgreen", "forestgreen", "green", "lime", "seagreen"]
        for _ in self.double_pendulums:
            (drag,) = self.ax.plot([], [], color=choice(shades_of_green))
            self.all_drags.append(drag)

    def init_rodes_and_points(self) -> List[Line2D]:
        """(Re)set every elements to zero.
        This allows to remove every elements from screen of time t, before printing/displaying new elements on time t+1

        Returns:
            List[Line2D]: Each elements, set to zero
        """
        # self.all_rodes_points is a list of tuple
        # self.all_rodes_points = [(rode1, point1, rode2, point2), ...]

        # all_rodes_points_detupled is the same list, without the tuple
        # all_rodes_points_detupled = [rode1, point1, rode2, point2, ...]
        all_rodes_points_detupled: List[Line2D] = []

        for rode_or_points in self.all_rodes_points:
            (rode1, point1, rode2, point2) = rode_or_points
            rode1.set_data([], [])
            point1.set_data([], [])
            rode2.set_data([], [])
            point2.set_data([], [])
            all_rodes_points_detupled.extend(rode_or_points)

        return all_rodes_points_detupled

    def init_drags(self) -> List[Line2D]:
        """(Re)set every drag to zero.
        This allows to remove every drag from screen of time t, before printing/displaying new drag on time t+1

        Returns:
            List[Line2D]: Each drag, set to zero
        """
        for drag in self.all_drags:
            drag.set_data([], [])

        # Here, self.all_drags is already a list of elements
        # (and not a list of tuple as self.all_rodes_points)
        # So, we can return it without further manipulation
        return self.all_drags

    def init(self) -> Tuple[Line2D, ...]:
        """(re)set every graphical elements to zero.
        This allows to remove them from screen of time t, before printing/displaying them on time t+1

        Returns:
            Tuple[Line2D, ...]: Each graphical elements reset
        """
        all_graphical_elements = []

        if self.with_drag:
            all_graphical_elements.extend(self.init_drags())

        all_graphical_elements.extend(self.init_rodes_and_points())

        return tuple(all_graphical_elements)

    def animate_rodes_point(self, t: int) -> List[Line2D]:
        """Animate the rodes and the points at time t

        Args:
            t (int): time

        Returns:
            List[Line2D]: List of rodes and points
        """
        # Same as in init_rodes_and_points
        # self.all_rodes_points is a list of tuple
        # self.all_rodes_points = [(rode1, point1, rode2, point2), ...]
        # all_rodes_points_detupled is the same list, without the tuple
        # all_rodes_points_detupled = [rode1, point1, rode2, point2, ...]
        all_rodes_points_detupled: List[Line2D] = []

        for dp, rode_or_points in zip(self.double_pendulums, self.all_rodes_points):
            # get the (x, y) coords of every object for the current pendulum at time t
            (pos_rode1, pos_point1, pos_rode2,
             pos_point2) = dp.get_pos_rodes_and_points(t)
            # get the rodes and points associated to this pendulum
            (rode1, point1, rode2, point2) = rode_or_points

            # set the position of the rodes and points
            rode1.set_data(*pos_rode1)
            point1.set_data(*pos_point1)
            rode2.set_data(*pos_rode2)
            point2.set_data(*pos_point2)

            all_rodes_points_detupled.extend(rode_or_points)

        return all_rodes_points_detupled

    def animate_drags(self, t: int) -> List[Line2D]:
        """Animate the drags at time t

        Args:
            t (int): time

        Returns:
            List[Line2D]: List of drags
        """
        for dp, drag in zip(self.double_pendulums, self.all_drags):
            drag.set_data(*dp.get_drag(t))

        return self.all_drags

    def animate(self, t: int) -> Tuple[Line2D, ...]:
        """Calculate and set the (x, y) coords of every graphical elements at time t.

        Args:
            t (int): time

        Returns:
            Tuple[Line2D, ...]: Each graphical elements reset
        """
        all_graphical_elements = []

        if self.with_drag:
            all_graphical_elements.extend(self.animate_drags(t))

        all_graphical_elements.extend(self.animate_rodes_point(t))

        return tuple(all_graphical_elements)

    def set_graph(self) -> None:
        """Set the title and the limits of the figure"""

        # set the title
        self.fig.suptitle("Double weighing pendulum")

        # calculate the max space needed to display every pendulum
        max_length = np.max([dp.l1 + dp.l2 for dp in self.double_pendulums])
        (x_min, x_max) = (-1.05 * max_length, 1.05 * max_length)
        (y_min, y_max) = (-1.05 * max_length, 1.05 * max_length)
        self.ax.axis([x_min, x_max, y_min, y_max])
        self.ax.set_aspect("equal", "box")  # to have a square figure


if __name__ == "__main__":
    first_dp = DoublePendulum([np.radians(0), np.radians(45), 0, 0])
    second_dp = DoublePendulum(
        [np.radians(0), np.radians(90), 0, 0], l1=2, l2=3)

    AnimateDoublePendulum([first_dp, second_dp])
