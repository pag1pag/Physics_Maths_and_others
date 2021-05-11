import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint

# plt.style.use('ggplot')


class WeighingPendulum:
    """
    yes
    """

    def __init__(
        self,
        X0,
        time_footstep=0.01,
        start_time=0,
        end_time=10,
        len_rode=1.0,
        gravity_acceleration=9.81,
    ):
        self.theta0 = self.check_theta0(X0)
        self.dt = time_footstep
        self.l = len_rode  # m
        self.g = gravity_acceleration  # m/s/s
        self.start_time = start_time
        self.end_time = end_time
        self.theta = self.resolve_equation()[:, 0]

    @staticmethod
    def check_theta0(x0):
        try:
            assert len(x0) == 2
        except TypeError:
            print(f"x0 should be an array, not {type(x0)}")
            quit()
        except AssertionError:
            print(f"x0 should be an array of lenght 2, not {x0}")
            quit()
        except:
            print("Oops! unexepected error")
            quit()

        try:
            assert 0 < x0[0] < np.pi
        except AssertionError:
            print(f"x0[0] should strictly be between 0 and pi radians, not {x0[0]}")
            quit()
        except:
            print("Oops! unexepected error")
            quit()

        return x0

    def equation_to_solve(self, X, t):
        """
        The derivative function
        """
        return np.array([X[1], -self.g / self.l * np.sin(X[0])])

    def get_time(self):
        return np.arange(self.start_time, self.end_time, self.dt)

    def resolve_equation(self):
        """
        resolve equation
        """
        T = self.get_time()
        return np.array(odeint(self.equation_to_solve, self.theta0, T))

    def get_coords(self):
        X = self.l * np.sin(self.theta)
        Y = -self.l * np.cos(self.theta)
        return X, Y

    def run_animation(self):
        fig, ax = plt.subplots()

        X, Y = self.get_coords()

        # windows limits
        (x_min, x_max) = (1.05 * np.min(X), 1.05 * np.max(X))
        (y_min, y_max) = (-1.05 * self.l, 1.05 * self.l)

        ax.axis([x_min, x_max, y_min, y_max])
        ax.set_aspect("equal", "box")

        # define the parameter
        (rode,) = ax.plot([], [], color="blue")
        (point,) = ax.plot([], [], ls="none", marker="o", color="red")
        ax.vlines(x=0, ymin=-self.l, ymax=0, color="k", ls="--")
        angle = ax.text(0, -0.5, "", ha="center", va="center", fontsize=12)

        def init():
            rode.set_data([], [])
            point.set_data([], [])
            return rode, point

        def animate(k):
            i = min(k, X.size)
            x_rode = [
                len_rode * X[i] for len_rode in np.arange(0, self.l, self.l / 100)
            ]
            y_rode = [
                len_rode * Y[i] for len_rode in np.arange(0, self.l, self.l / 100)
            ]

            rode.set_data(x_rode, y_rode)
            point.set_data(X[i], Y[i])

            angle.set_text(f"{np.rad2deg(self.theta[i]):.1f}")

            return rode, point, angle

        # pylint: disable=unused-variable
        ani = animation.FuncAnimation(
            fig=fig,
            func=animate,
            init_func=init,
            frames=range(X.size),
            interval=50,
            blit=True,
        )

        plt.show()


if __name__ == "__main__":
    X0 = [np.radians(150), 0]
    # X0 = 5
    my_pendulum = WeighingPendulum(X0)
    my_pendulum.run_animation()
