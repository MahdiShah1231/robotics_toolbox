from abc import ABC, abstractmethod

import numpy
import numpy as np
from matplotlib import pyplot as plt


class TrajectoryBase(ABC):
    """Abstract Base Class for a trajectory generator."""

    def __init__(self) -> None:
        """Initialise the base trajectory generator."""
        self._waypoints = ()
        self._traj_time = None
        self._frequency = None
        self._t_values = []
        self._setpoints = []

    def setup_trajectory(self,
                         waypoints: tuple[float, float],
                         time: float = 3.0,
                         frequency: int = 100) -> None:
        """Set up the trajectory waypoints.

        Args:
            waypoints: The trajectory waypoints (x,y), start and finish.
            time: The time taken to complete the trajectory.
            frequency: The sampling frequency of the trajectory points.
        """
        self._clear_traj()  # First clear any existing solutions
        self._waypoints = waypoints
        self._traj_time = time
        self._frequency = frequency

    @abstractmethod
    def solve_traj(self) -> tuple[numpy.ndarray, list[float]]:
        """Abstract method to be implemented."""
        raise NotImplementedError

    def _clear_traj(self) -> None:
        """Clear the saved trajectory."""
        self._t_values = []
        self._setpoints = []

    def plot_traj(self) -> None:
        """Plot the saved trajectory."""
        fig, ax = plt.subplots()
        ax.set_xlim([0, self._traj_time])
        ax.plot(self._t_values, self._setpoints)
        plt.show()


class QuinticPolynomialTrajectory(TrajectoryBase):
    """A derived trajectory generator to create a quintic polynomial trajectory.

    Used to solve trajectories with 6 constraints on starting and final positions, velocities and accelerations.

    """
    # a0 + a1t + a2t^2 + a3t^3 + a4t^4 + a5t^5 = final_position
    # 0a0 + a1 + 2a2t + 3a3t^2 + 4a4t^3 + 5a5t^4 = final_velocity = 0
    # 0a0 + 0a1 + 2a2 + 6a3t + 12a4t^2 + 20a5t^3 = final_acceleration = 0
    # a0 + 0 + 0 + 0 + 0 + 0 = initial position
    # 0 + a1 + 0 + 0 + 0 + 0 = initial velocity = 0
    # 0 + 0 + 2a2 + 0 + 0 + 0 = initial_acceleration = 0
    # Ax = b

    def __init__(self) -> None:
        """Initialise the quintic polynomial generator."""
        super().__init__()

    def solve_traj(self) -> tuple[numpy.ndarray[float], list[float]]:
        """Solve the quintic polynomial trajectory.

        Returns:
            A tuple containing the sampled time points and the corresponding setpoints.
        """
        setpoints = []
        # Ax = b, solving for x (matrix of coefficients a0 -> a5) to give trajectory going from initial setpoint
        # to final setpoint in t = traj_time
        a_matrix = np.array([[1, self._traj_time, self._traj_time ** 2, self._traj_time ** 3, self._traj_time ** 4, self._traj_time ** 5],
                             [0, 1, 2 * self._traj_time, 3 * (self._traj_time ** 2), 4 * (self._traj_time ** 3), 5 * (self._traj_time ** 4)],
                             [0, 0, 0, 6 * self._traj_time, 12 * (self._traj_time ** 2), 20 * (self._traj_time ** 3)],
                             [1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 2, 0, 0, 0]])

        b_matrix = np.array([self._waypoints[1], 0, 0, self._waypoints[0], 0, 0])

        x_matrix = np.linalg.solve(a_matrix, b_matrix)

        # Now need to evaluate the coefficients for 0 < t < traj_time at the increments specified for the whole trajectory
        # a0 + a1t + a2t^2 + a3t^3 + a4t^4 + a5t^5 = final_position for t = traj time, = intermediate position for t < traj_time

        t_values = np.linspace(0, self._traj_time, self._frequency)

        for t in t_values:
            setpoint = x_matrix[0] + x_matrix[1] * t + x_matrix[2] * (t**2) + x_matrix[3] * (t**3) + x_matrix[4] * (t**4) + x_matrix[5] * (t**5)
            setpoints.append(setpoint)

        self._t_values = t_values
        self._setpoints = setpoints

        return t_values, setpoints


class LinearTrajectory(TrajectoryBase):
    def __init__(self) -> None:
        super().__init__()

    def solve_traj(self) -> tuple[numpy.ndarray, list[float]]:
        raise NotImplementedError


class ParabolicBlendTrajectory:
    def __init__(self) -> None:
        super().__init__()

    def solve_traj(self) -> tuple[numpy.ndarray, list[float]]:
        raise NotImplementedError


class RuckigTrajectory:
    def __init__(self) -> None:
        super().__init__()

    def solve_traj(self) -> tuple[numpy.ndarray, list[float]]:
        raise NotImplementedError


if __name__ == '__main__':

    # Testing Quintic Poly Traj
    quintic_poly_traj = QuinticPolynomialTrajectory()
    quintic_poly_traj.setup_trajectory(waypoints=(0.0, 0.2), time=3.0, frequency=100)
    quintic_poly_traj.solve_traj()
    quintic_poly_traj.plot_traj()

