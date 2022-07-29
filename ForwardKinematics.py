import numpy as np
from helper_functions.helper_functions import calculate_joint_angles, wrap_angles_to_pi


# Remove DH params, if the robots already been created then we don't need to do any extra things?

# In forward kinematics mode, cant control the linear base, only enter link configs. therefore len(targ) = len(joint_config)-1 if linear base is on


class ForwardKinematics:
    def __init__(self, robot, target_configuration):
        self.robot = robot
        self.linear_base = robot.linear_base
        self.link_num = robot.link_number
        self.joint_configuration = robot.joint_configuration
        self.link_lengths = robot.link_lengths
        self.vertices = robot.vertices
        self.target_configuration = wrap_angles_to_pi(target_configuration)

    def forward_kinematics(self):
        if self.linear_base:
            self.target_configuration.insert(0, 0.0)
        reference_angle = 0
        for link_index in range(self.link_num):
            self.vertices["x"][link_index + 1] = self.vertices["x"][link_index] + self.link_lengths[link_index] * np.cos(self.target_configuration[link_index] + reference_angle)
            self.vertices["y"][link_index + 1] = self.vertices["y"][link_index] + self.link_lengths[link_index] * np.sin(self.target_configuration[link_index] + reference_angle)
            reference_angle += self.target_configuration[link_index]
        self.joint_configuration = calculate_joint_angles(self.vertices)
        print("Final robot configuration:")
        print(self.joint_configuration)
        print(self.vertices)
