from typing import Union
import numpy as np
from robot import Robot
from inverse_kinematics import FabrikSolver
from helper_functions.helper_functions import MoveType, create_logger
import logging

logger = create_logger(__name__, logging.INFO)

def create_robot(link_lengths: list[float],
                 ik_alg=FabrikSolver(),
                 joint_configuration: list[float] = None,
                 robot_base_radius: float = None,
                 linear_base: bool = False,
                 environment: Union[str, list[float]] = None) -> Robot:

    robot = Robot(link_lengths=link_lengths, ik_solver=ik_alg, joint_configuration=joint_configuration,
                  robot_base_radius=robot_base_radius, linear_base=linear_base, environment=environment)
    return robot


def move(robot: Robot,
         move_type: MoveType,
         enable_animation: bool = True,
         **kwargs) -> None:
    robot.move(move_type=move_type,
               plot=True,
               enable_animation=enable_animation,
               **kwargs)
    logger.info(f"Final joint configuration: {robot.joint_configuration}")
    logger.info(f"Final vertices: {robot.vertices}")

if __name__ == '__main__':
    r = create_robot(link_lengths=[0.4, 0.3, 0.2],
                     ik_alg=FabrikSolver(),
                     joint_configuration=None,
                     robot_base_radius=0.1,
                     linear_base=True,
                     environment=None)
    joint_space_move = MoveType.JOINT
    cartesian_space_move = MoveType.CARTESIAN

    # Joint space movements using wrapper function
    # Pass move_type = MoveType.JOINT, and send a keyword argument "target_configuration"

    # Example joint space cmd, uncomment below
    # move(robot=r, move_type=joint_space_move, target_configuration=[1.57,1.57,0])

    # Cartesian space movements using wrapper function
    # Pass move_type = MoveType.CARTESIAN, and send keyword argument "target_position" and "target_orientation"

    # Example cartesian space cmd, uncomment below
    move(robot=r, move_type=cartesian_space_move, target_position=[1.5,0.6], target_orientation=np.pi / 2, mirror=False)

