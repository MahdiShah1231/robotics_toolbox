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

def forward_kinematics():
    pass



if __name__ == '__main__':
    r = create_robot(link_lengths=[0.4, 0.3, 0.2], ik_alg=Fabrik, robot_base_radius=0.1, linear_base=True)
    inverse_kinematics(r, [0.8, 0.6], np.pi / 2, None, True, False)