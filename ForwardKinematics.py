import numpy as np
from helper_functions.helper_functions import calculate_joint_angles, wrap_angles_to_pi


class ForwardKinematics:
    def __init__(self, robot, target_configuration):
        self.__robot = robot
        self.__target_configuration = wrap_angles_to_pi(target_configuration)

    def forward_kinematics(self, initialise=False):
        if self.__robot.linear_base:
            self.__target_configuration.insert(0, 0.0)
        reference_angle = 0
        for link_index in range(self.__robot.n_links):
            self.__robot.vertices["x"][link_index + 1] = self.__robot.vertices["x"][link_index] + \
                                                   self.__robot.link_lengths[link_index] * \
                                                   np.cos(self.__target_configuration[link_index] + reference_angle)

            self.__robot.vertices["y"][link_index + 1] = self.__robot.vertices["y"][link_index] + \
                                                         self.__robot.link_lengths[link_index] * \
                                                         np.sin(self.__target_configuration[link_index] + reference_angle)

            reference_angle += self.__target_configuration[link_index]
        self.__robot.joint_configuration = calculate_joint_angles(self.__robot.vertices)
        if not initialise:
            print("Final robot configuration:")
            print(self.__robot.joint_configuration)
            print(self.__robot.vertices)
