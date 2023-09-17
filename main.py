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
    """Create a Robot object with the given parameters.

         Args:
             link_lengths: Link lengths for the robot. Length of the list = number of links.
             ik_solver: IK Solver for the Inverse Kinematics calculations.
             trajectory_generator: Trajectory generator for the animated motions.
             joint_configuration: Starting joint configuration. OPTIONAL.
             robot_base_radius: Radius of the circular base.
             linear_base: Enables/disables use of a linear rail to move along horizontal axis.
             environment: To be implemented.

         Returns:
             An initialised Robot object.
    """
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
    """Move the robot either in cartesian or joint space.

        Args:
            robot: An initialised Robot object.
            move_type: MoveType enum specifying the coordinate space of the motion command.
            enable_animation: Dictates whether to display animated motions or snap into place.
    """
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

    # Motion types
    # MoveType.JOINT  = Joint Space Motion (Forward Kinematics)
    # MoveType.CARTESIAN = Cartesian Space Motion (Inverse Kinematics)

    # Insert Motion Commands below using move()

