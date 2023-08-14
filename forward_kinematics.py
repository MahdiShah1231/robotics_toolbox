import numpy as np
from helper_functions.helper_functions import calculate_joint_angles, wrap_angles_to_pi
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot import Robot


class ForwardKinematics:
    def __init__(self, robot: 'Robot', target_configuration: list[float]) -> None:
        self.__robot = robot
        self.__target_configuration = wrap_angles_to_pi(target_configuration)

    def forward_kinematics(self, debug: bool) -> dict[str, list[float]]:
        # Need to dummy joint angle for the linear prismatic joint to avoid indexing error
        start_idx = 0
        if self.__robot.linear_base:
            start_idx = 1  # Skipping first link since the prismatic joint wont be controlled through FK
            self.__target_configuration.insert(0, 0.0)

        reference_angle = 0
        for link_index in range(start_idx, self.__robot.n_links):
            # Joint angles are computed as relative to previous link (convention)
            behind_vertex = [self.__robot.vertices["x"][link_index], self.__robot.vertices["y"][link_index]]
            angle = self.__target_configuration[link_index] + reference_angle
            x_change = self.__robot.link_lengths[link_index] * np.cos(angle)
            y_change = self.__robot.link_lengths[link_index] * np.sin(angle)

            new_vertex = np.add(behind_vertex, [x_change, y_change])

            self.__robot.vertices["x"][link_index + 1] = new_vertex[0]
            self.__robot.vertices["y"][link_index + 1] = new_vertex[1]

            reference_angle += self.__target_configuration[link_index]
        self.__robot.joint_configuration = calculate_joint_angles(self.__robot.vertices)
        if debug:
            print("Final robot configuration:")
            print(self.__robot.joint_configuration)
            print(self.__robot.vertices)

        return self.__robot.vertices
