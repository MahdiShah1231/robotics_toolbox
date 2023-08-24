from typing import Union
import numpy as np
from robot import Robot
from inverse_kinematics import Fabrik
from helper_functions.helper_functions import MoveType


def create_robot(link_lengths: list[float],
                 ik_alg=None,
                 joint_configuration: list[float] = None,
                 robot_base_radius: float = None,
                 linear_base: bool = False,
                 environment: Union[str, list[float]] = None) -> Robot:

    robot = Robot(link_lengths=link_lengths,
                  ik_alg=ik_alg,
                  joint_configuration=joint_configuration,
                  robot_base_radius=robot_base_radius,
                  linear_base=linear_base,
                  environment=environment)
    return robot


def inverse_kinematics(robot: Robot,
                       target_position: list[float],
                       target_orientation: float = None,
                       mirror: bool = True,
                       debug: bool = False,
                       plot: bool = True) -> None:

    robot.inverse_kinematics(target_position=target_position,
                             target_orientation=target_orientation,
                             mirror=mirror,
                             debug=debug,
                             plot=plot)


def move(robot: Robot,
         move_type: MoveType,
         debug: bool = False,
         enable_animation: bool = True,
         **kwargs) -> None:
    robot.move(move_type=move_type,
               plot=True,
               enable_animation=enable_animation,
               debug=debug,
               **kwargs)

if __name__ == '__main__':
    r = create_robot(link_lengths=[0.4, 0.3, 0.2],
                     ik_alg=Fabrik,
                     joint_configuration=None,
                     robot_base_radius=0.1,
                     linear_base=True,
                     environment=None)
    # inverse_kinematics(robot=r, target_position=[0.5, 0.6], target_orientation=np.pi / 2, mirror=True, debug=False)

    joint_space_move = MoveType.JOINT
    cartesian_space_move = MoveType.CARTESIAN

    # Joint space movements using wrapper function
    # Pass move_type = MoveType.JOINT, and send a keyword argument "target_configuration"
    move(robot=r, move_type=joint_space_move, target_configuration=[1.57,1.57,0])
