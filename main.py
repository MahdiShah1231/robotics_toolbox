import numpy as np
from Robot import Robot
from InverseKinematics import Fabrik

def create_robot(link_lengths,
                 ik_alg=None,
                 joint_configuration=None,
                 robot_base_radius=None,
                 linear_base=False,
                 environment=None):

    robot = Robot(link_lengths=link_lengths,
                  ik_alg=ik_alg,
                  joint_configuration=joint_configuration,
                  robot_base_radius=robot_base_radius,
                  linear_base=linear_base,
                  environment=environment)
    return robot


def inverse_kinematics(robot,
                       target_position,
                       target_orientation=None,
                       mirror=True,
                       debug=False,
                       plot=False):

    robot.inverse_kinematics(target_position=target_position,
                             target_orientation=target_orientation,
                             mirror=mirror,
                             debug=debug,
                             plot=plot)


def forward_kinematics(robot, target_configuration, debug=False, plot=False):
    robot.forward_kinematics(target_configuration=target_configuration, debug=debug, plot=plot)


if __name__ == '__main__':
    r = create_robot(link_lengths=[0.4, 0.3, 0.2],
                     ik_alg=Fabrik,
                     joint_configuration=None,
                     robot_base_radius=0.1,
                     linear_base=True,
                     environment=None)
    # inverse_kinematics(robot=r, target_position=[1, 0.6], target_orientation=np.pi / 2, mirror=True, debug=True)
    # forward_kinematics(robot=r, target_configuration=[0, np.pi/2, -np.pi/2])
    # forward_kinematics(robot=r, target_configuration=[0, 0, 0])
