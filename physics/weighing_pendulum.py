from typing import List, Tuple

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import odeint


class Pendulum:
    """This class implements the physics behind the simple weighing pendulum"""

    def __init__(
        self,
        initial_conditions: List[float],
        time_step: float = 0.01,
        end_time: float = 10,
        len_rode: float = 1.0,
        gravity_acceleration: float = 9.81,
    ):
        """Init function

        Args:
            initial_conditions (List[float]): A list of 2 elements.
                The 1st is the initial angle between the rode and the -y-axis [rad].
                The 2nd is the initial angular velocity [rad.s^-1].
            time_step (float, optional): timestep of simulation [s]. Defaults to 0.01.
            end_time (float, optional): duration of the simulation [s]. Defaults to 10.
            len_rode (float, optional): length of the rode [m]. Defaults to 1.0.
            gravity_acceleration (float, optional): gravitational constant [m.s^-2]. Defaults to 9.81.
        """
        self.initial_conditions = self.check_ic(initial_conditions)
        self.l = len_rode  # m
        self.g = gravity_acceleration  # m.s^-2
        self.end_time = end_time  # s
        self.times = np.arange(0, end_time, time_step)

        self.theta = self.solve_equation()[:, 0]
        x, y = self.get_coords()
        self.x = x
        self.y = y

    @staticmethod
    def check_ic(initial_conditions: List[float]) -> List[float]:
        """Check the initial conditions

        Args:
            initial_conditions (List[float]): [theta0, thetadot0]

        Raises:
            ValueError: initial_conditions should be a list of 2 elements
            ValueError: theta0 should be between -pi and pi

        Returns:
            List[float]: [theta0, thetadot0]
        """
        if not isinstance(initial_conditions, list):
            raise ValueError("initial_conditions should be a list")
        if len(initial_conditions) != 2:
            raise ValueError("Should have 2 elements in initial_conditions")
        if not -np.pi < initial_conditions[0] < np.pi:
            raise ValueError(
                f"The initial angle should be between -pi and pi radians, not {initial_conditions[0]}")
        return initial_conditions

    def f(self, x, _) -> np.ndarray:
        """Ordinary differential equation: Xdot = f(X, t), with:
        - X[0] = theta
        - X[1] = thetadot

        - Xdot[0] = thetadot = X[1]
        - Xdot[1] = thetadotdot = -g/l * sin(theta) = -g/l * sin(X[0])
        """
        return np.array([x[1], -self.g / self.l * np.sin(x[0])])

    def solve_equation(self) -> np.ndarray:
        """Solve the ODE and return the solution.
        The solution is an array of shape: (len(self.times), 2)
        """
        return np.array(odeint(self.f, self.initial_conditions, self.times))

    def get_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the coordinate of the point

        Returns:
            Tuple[np.ndarray, np.ndarray]: (x, y) coords
        """
        x = self.l * np.sin(self.theta)
        y = -self.l * np.cos(self.theta)
        return x, y

    def get_pos_rodes_and_points(self, t: int) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
        """
        Calculate the position of the two points and the two rodes at time t.

        Args:
            - t (int): instant/frame to display

        Returns:
            The (x, y) coordinates of the two points, and the list of (x, y) coordinates describing the rodes, at time t
        """
        x_rode = np.array([len_rode * self.x[t]/self.l
                           for len_rode in np.arange(0, self.l, self.l / 100)])
        y_rode = np.array([len_rode * self.y[t]/self.l
                           for len_rode in np.arange(0, self.l, self.l / 100)])

        rode = (x_rode, y_rode)
        point = (self.x[t], self.y[t])

        return (rode, point)


class AnimatePendulum:
    """A class to animate multiple simple pendulum on the same graph."""

    def __init__(self, pendulums: List[Pendulum], save=False):
        if not isinstance(pendulums, list):
            raise ValueError("pendulums should be a list")
        for dp in pendulums:
            if not isinstance(dp, Pendulum):
                raise ValueError(
                    "pendulums should be a list of Pendulum")
        self.pendulums = pendulums

        # The figure on which to display the animation
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.set_graph()

        # -- The graphical elements to display on the figure --#
        # This corresponds to the points (in red), and the lines (in red)
        self.all_rodes_points: List[Tuple[Line2D, ...]] = []
        self.create_rodes_and_points()

        self.run_animation(save=save)

    def set_graph(self) -> None:
        """Set the title and the limits of the figure"""

        # set the title
        self.fig.suptitle("Weighing pendulum")

        # calculate the max space needed to display every pendulum
        max_length = np.max([pendulum.l for pendulum in self.pendulums])
        (x_min, x_max) = (-1.05 * max_length, 1.05 * max_length)
        (y_min, y_max) = (-1.05 * max_length, 1.05 * max_length)
        self.ax.axis([x_min, x_max, y_min, y_max])
        self.ax.set_aspect("equal", "box")  # to have a square figure

    def create_rodes_and_points(self) -> None:
        """For each pendulum, create 2 elements (1 rode and 1 point).
        Rodes are blue, points are red (sugar is sweet, and so are you).

        Here, we only create the elements, but they don't have data inside.
        """
        for _ in self.pendulums:
            (rode,) = self.ax.plot([], [], color="blue")
            (point,) = self.ax.plot([], [], ls="none", marker="o", color="red")

            # self.all_rodes_points is a list of tuple
            self.all_rodes_points.append((rode, point))

    def init_rodes_and_points(self) -> List[Line2D]:
        """(Re)set every elements to zero.
        This allows to remove every elements from screen of time t, before printing/displaying new elements on time t+1

        Returns:
            List[Line2D]: Each elements, set to zero
        """
        # self.all_rodes_points is a list of tuple
        # self.all_rodes_points = [(rode1, point1), (rode2, point2), ...]

        # all_rodes_points_detupled is the same list, without the tuple
        # all_rodes_points_detupled = [rode1, point1, rode2, point2, ...]
        all_rodes_points_detupled: List[Line2D] = []

        for rode_or_points in self.all_rodes_points:
            (rode, point) = rode_or_points
            rode.set_data([], [])
            point.set_data([], [])
            all_rodes_points_detupled.extend(rode_or_points)

        return all_rodes_points_detupled

    def init(self) -> Tuple[Line2D, ...]:
        """(re)set every graphical elements to zero.
        This allows to remove them from screen of time t, before printing/displaying them on time t+1

        Returns:
            Tuple[Line2D, ...]: Each graphical elements reset
        """
        all_graphical_elements = []

        # if self.with_drag:
        #     all_graphical_elements.extend(self.init_drags())

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
        # self.all_rodes_points = [(rode1, point1), (rode2, point2), ...]
        # all_rodes_points_detupled is the same list, without the tuple
        # all_rodes_points_detupled = [rode1, point1, rode2, point2, ...]
        all_rodes_points_detupled: List[Line2D] = []

        for pendulum, rode_or_points in zip(self.pendulums, self.all_rodes_points):
            # get the (x, y) coords of every object for the current pendulum at time t
            (pos_rode, pos_point) = pendulum.get_pos_rodes_and_points(t)
            # get the rodes and points associated to this pendulum
            (rode, point) = rode_or_points

            # set the position of the rodes and points
            rode.set_data(*pos_rode)
            point.set_data(*pos_point)

            all_rodes_points_detupled.extend(rode_or_points)

        return all_rodes_points_detupled

    def animate(self, t: int) -> Tuple[Line2D, ...]:
        """Calculate and set the (x, y) coords of every graphical elements at time t.

        Args:
            t (int): time

        Returns:
            Tuple[Line2D, ...]: Each graphical elements reset
        """
        all_graphical_elements = []

        # if self.with_drag:
        #     all_graphical_elements.extend(self.animate_drags(t))
        #  angle.set_text(f"{np.rad2deg(self.theta[i]):.1f}")

        all_graphical_elements.extend(self.animate_rodes_point(t))

        return tuple(all_graphical_elements)

    def run_animation(self, save: bool = False) -> None:
        """The main function of this class"""
        # run the simulation only over the smallest period of time
        # (if they are double pendulum with different simulation time, choose the smallest of this time)
        min_simulation_time = np.min(
            [pendulum.times.size for pendulum in self.pendulums])

        anim = animation.FuncAnimation(
            fig=self.fig,
            func=self.animate,
            frames=range(min_simulation_time),
            init_func=self.init,
            interval=5,  # Delay between frames in milliseconds.
            blit=True,
        )

        if save:
            anim.save("weighing_pendulum.mp4")
        else:
            # run the animation
            plt.show()


if __name__ == "__main__":
    first_pendulum = Pendulum([np.radians(150), 0])
    second_pendulum = Pendulum([np.radians(-150), 0])
    AnimatePendulum([first_pendulum, second_pendulum])
    # my_pendulum.run_animation()
