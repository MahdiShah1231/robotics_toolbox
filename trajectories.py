from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt


class BaseTrajectoryClass(ABC):
    def __init__(self, waypoints: tuple, time: float, frequency: int):
        self.waypoints = waypoints
        self.traj_time = time
        self.frequency = frequency
        self.t_values = []
        self.setpoints = []

    @abstractmethod
    def solve_traj(self):
        raise NotImplementedError

    def clear_traj(self):
        self.t_values = []
        self.setpoints = []

    def plot_traj(self):
        fig, ax = plt.subplots()
        ax.set_xlim([0, self.traj_time])
        ax.plot(self.t_values, self.setpoints)
        plt.show()


class QuinticPolynomialTrajectory(BaseTrajectoryClass):
    # a0 + a1t + a2t^2 + a3t^3 + a4t^4 + a5t^5 = final_position
    # 0a0 + a1 + 2a2t + 3a3t^2 + 4a4t^3 + 5a5t^4 = final_velocity = 0
    # 0a0 + 0a1 + 2a2 + 6a3t + 12a4t^2 + 20a5t^3 = final_acceleration = 0
    # a0 + 0 + 0 + 0 + 0 + 0 = initial position
    # 0 + a1 + 0 + 0 + 0 + 0 = initial velocity = 0
    # 0 + 0 + 2a2 + 0 + 0 + 0 = initial_acceleration = 0
    # Ax = b
    def __init__(self, waypoints: tuple, time: float = 3.0, frequency: int = 100):
        super().__init__(waypoints, time, frequency)

    def solve_traj(self):
        setpoints = []
        # Ax = b, solving for x (matrix of coefficients a0 -> a5) to give trajectory going from initial setpoint
        # to final setpoint in t = traj_time
        a_matrix = np.array([[1, self.traj_time, self.traj_time**2, self.traj_time**3, self.traj_time**4, self.traj_time**5],
                             [0, 1, 2*self.traj_time, 3*(self.traj_time**2), 4*(self.traj_time**3), 5*(self.traj_time**4)],
                             [0, 0, 0, 6*self.traj_time, 12*(self.traj_time**2), 20*(self.traj_time**3)],
                             [1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 2, 0, 0, 0]])

        b_matrix = np.array([self.waypoints[1], 0, 0, self.waypoints[0], 0, 0])

        x_matrix = np.linalg.solve(a_matrix, b_matrix)

        # Now need to evaluate the coefficients for 0 < t < traj_time at the increments specified for the whole trajectory
        # a0 + a1t + a2t^2 + a3t^3 + a4t^4 + a5t^5 = final_position for t = traj time, = intermediate position for t < traj_time

        t_values = np.linspace(0, self.traj_time, self.frequency)

        for t in t_values:
            setpoint = x_matrix[0] + x_matrix[1] * t + x_matrix[2] * (t**2) + x_matrix[3] * (t**3) + x_matrix[4] * (t**4) + x_matrix[5] * (t**5)
            setpoints.append(setpoint)

        self.t_values = t_values
        self.setpoints = setpoints

        return t_values, setpoints


class LinearTrajectory(BaseTrajectoryClass):
    def __init__(self, waypoints: tuple, time: float = 3.0, frequency: int = 100):
        super().__init__(waypoints, time, frequency)

    def solve_traj(self):
        raise NotImplementedError


class ParabolicBlendTrajectory:
    def __init__(self, waypoints: tuple, time: float = 3.0, frequency: int = 100):
        super().__init__(waypoints, time, frequency)

    def solve_traj(self):
        raise NotImplementedError


class RuckigTrajectory:
    def __init__(self, waypoints: tuple, time: float = 3.0, frequency: int = 100):
        super().__init__(waypoints, time, frequency)

    def solve_traj(self):
        raise NotImplementedError


if __name__ == '__main__':

    # Testing Quintic Poly Traj
    quintic_poly_traj = QuinticPolynomialTrajectory((0.0, 0.2))
    quintic_poly_traj.solve_traj()
    quintic_poly_traj.plot_traj()

