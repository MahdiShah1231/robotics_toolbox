from typing import List, Optional, Any

import numpy as np
from Robot import Robot
from InverseKinematics import Fabrik
import matplotlib.pyplot as plt


def create_robot(link_lengths: List[float],
                 ik_alg: Optional[Any] = None,
                 joint_configuration: Optional[List[float]] = None,
                 robot_base_radius: Optional[float] = None,
                 linear_base: bool = False,
                 environment: Any = None) -> Robot:

    robot = Robot(link_lengths=link_lengths,
                  ik_alg=ik_alg,
                  joint_configuration=joint_configuration,
                  robot_base_radius=robot_base_radius,
                  linear_base=linear_base,
                  environment=environment)
    return robot


def inverse_kinematics(robot: Robot,
                       target_position: List[float],
                       target_orientation: Optional[float] = None,
                       mirror: bool = True,
                       debug: bool = False,
                       plot: bool = True) -> None:

    if plot:
        fig, ax = plt.subplots()
        robot.inverse_kinematics(target_position=target_position,
                                 target_orientation=target_orientation,
                                 mirror=mirror,
                                 debug=debug,
                                 ax=ax)


def forward_kinematics(robot: Robot,
                       target_configuration: List[float],
                       debug: bool = False,
                       plot: bool = True) -> None:
    if plot:
        fig, ax = plt.subplots()
        robot.forward_kinematics(target_configuration=target_configuration, debug=debug, ax=ax)


if __name__ == '__main__':
    r = create_robot(link_lengths=[0.4, 0.3, 0.2],
                     ik_alg=Fabrik,
                     joint_configuration=None,
                     robot_base_radius=0.1,
                     linear_base=True,
                     environment=None)
    inverse_kinematics(robot=r, target_position=[0.5, 0.6], target_orientation=np.pi / 2, mirror=True, debug=False)
    # forward_kinematics(robot=r, target_configuration=[0, np.pi/2, -np.pi/2], debug=False)
