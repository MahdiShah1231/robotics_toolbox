import numpy as np

from Robot import Robot
from InverseKinematics import *


def create_robot(link_lengths, ik_alg=None, joint_configuration=None, robot_base_radius=None, linear_base=False):
    robot = Robot(link_lengths=link_lengths,
                  ik_alg=ik_alg,
                  joint_configuration=joint_configuration,
                  robot_base_radius=robot_base_radius,
                  linear_base=linear_base)
    return robot

def inverse_kinematics(robot, target_position, target_orientation=None, environment=None, mirror=True, debug=False):
    robot.inverse_kinematics(target_position=target_position,
                             target_orientation=target_orientation,
                             environment=environment,
                             mirror=mirror,
                             debug=debug)

def forward_kinematics(robot, target_configuration):
    robot.forward_kinematics(target_configuration=target_configuration)



if __name__ == '__main__':
    r = create_robot(link_lengths=[0.4, 0.3, 0.2], ik_alg=Fabrik, joint_configuration=None, robot_base_radius=0.1, linear_base=True)
    # inverse_kinematics(robot=r, target_position=[0.6, 0.6], target_orientation=np.pi / 2, environment=None, mirror=True, debug=False)
    forward_kinematics(robot=r, target_configuration=[0, np.pi/2, 0])