from typing import Union
import numpy as np
from robot import Robot
from inverse_kinematics import IKSolverBase, FabrikSolver
from helper_functions.helper_functions import MoveType, create_logger
from trajectories import TrajectoryBase, QuinticPolynomialTrajectory
import logging

logger = create_logger(__name__, logging.INFO)


def create_robot(link_lengths: list[float],
                 ik_solver: IKSolverBase = FabrikSolver(),
                 trajectory_generator: TrajectoryBase = QuinticPolynomialTrajectory(),
                 joint_configuration: list[float] = None,
                 robot_base_radius: float = None,
                 linear_base: bool = False,
                 environment: Union[str, list[float]] = None) -> Robot:

    robot = Robot(link_lengths=link_lengths,
                  ik_solver=ik_solver,
                  trajectory_generator=trajectory_generator,
                  joint_configuration=joint_configuration,
                  robot_base_radius=robot_base_radius,
                  linear_base=linear_base,
                  environment=environment)

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

    # Change these parameters as needed
    robot_link_lengths = [0.4, 0.3, 0.2]
    robot_base_radius = 0.1
    starting_joint_configuration = None
    linear_base = True
    ik_solver = FabrikSolver()
    trajectory_generator = QuinticPolynomialTrajectory()

    r = create_robot(link_lengths=robot_link_lengths,
                     ik_solver=ik_solver,
                     trajectory_generator=trajectory_generator,
                     joint_configuration=starting_joint_configuration,
                     robot_base_radius=robot_base_radius,
                     linear_base=linear_base,
                     environment=None)

    joint_space_move = MoveType.JOINT
    cartesian_space_move = MoveType.CARTESIAN

    # Joint space movements using wrapper function
    # Pass move_type = MoveType.JOINT, and send a keyword argument "target_configuration"

    # Example joint space cmd, uncomment below
    move(robot=r, move_type=joint_space_move, target_configuration=[1.57,1.57,0])

    # Cartesian space movements using wrapper function
    # Pass move_type = MoveType.CARTESIAN, and send keyword argument "target_position" and "target_orientation"

    # Example cartesian space cmd, uncomment below
    move(robot=r, move_type=cartesian_space_move, target_position=[1.5,0.6], target_orientation=np.pi / 2, mirror=False)

