import logging
from typing import Union
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum


def create_logger(module_name: str, level: int) -> logging.Logger:
    """Create a logger.

    Args:
        module_name: The __name__ property of the calling module.
        level: The desired logging level of the logger.

    Returns:
        A logging.Logger object.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    logger_handler = logging.StreamHandler()
    logger_formatter = logging.Formatter('[%(levelname)s][%(name)s]: %(message)s')
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)

    return logger


class MoveType(Enum):
    """Enum class for specifying coordinate space for move commands."""
    JOINT = "Joint"  # Joint space command, Forward Kinematics
    CARTESIAN = "Cartesian"  # Cartesian space command, Inverse Kinematics


def calculate_joint_angles(vertices: dict[str, list[float]], linear_base: bool) -> list[float]:
    """Calculate joint angles given the vertices of the robot and active state of linear rail.

    Args:
        vertices: The current vertices.
        linear_base: The active state of the linear rail.

    Returns:
        A list of floats giving the joint angles of the robot.
    """
    joint_angles = [0.0] * (len(vertices["x"]) - 1)
    old_direction_vector = [1, 0]  # Starting reference vector (taking angle from positive x)
    last_vertex_index = len(vertices["x"]) - 1
    start_vertex_index = 0 if not linear_base else 1

    # Looping through vertices to calculate angles between the links using linear algebra
    for vertex_index in range(start_vertex_index, last_vertex_index):
        vertex = [vertices["x"][vertex_index], vertices["y"][vertex_index]]
        vertex_ahead = [vertices["x"][vertex_index + 1], vertices["y"][vertex_index + 1]]
        new_direction_vector = np.subtract(vertex_ahead, vertex)
        new_direction_vector = new_direction_vector / np.linalg.norm(new_direction_vector)

        # in 2D, cross product of 2 vectors: A X B = |A|*|B|*Sin(theta)
        # Dot product: A . B = |A|*|B|*Cos(theta)
        # Tan(theta) = Sin(theta)/Cos(theta)
        # Therefore signed clockwise angle between vectors is given as Atan2(cross_prod, dot_prod)
        cross_prod = np.cross(old_direction_vector, new_direction_vector)
        dot_prod = np.dot(old_direction_vector, new_direction_vector)
        joint_angle = np.arctan2(cross_prod, dot_prod)
        joint_angles[vertex_index] = joint_angle

        old_direction_vector = new_direction_vector  # Updating the reference vector

    return joint_angles


def wrap_angle_to_pi(angle: float) -> float:
    """Wrap angle between -pi and pi.

    Args:
        angle: The angle to be wrapped.

    Returns:
        A float of the wrapped angle.
    """
    # Wrap angle if angle > pi
    if angle > np.pi:

        # Floor div to find int number of times the angle wraps around pi
        wrap_count = angle // np.pi

        # If wrap_count is a multiple of 2 (e.g angle = 2pi, 4pi, ... etc),
        # angle goes through integer full rotations of 2pi
        if wrap_count % 2 == 0:
            # Subtract pi * wrap_count from the angle to unwind it through n=wrap_count full rotations
            wrapped_angle = (angle - (np.pi * wrap_count))

        # If wrap_count % 2 != 0 here (including angle > np.pi) then the angle consists of multiples of a whole rotation
        # plus some remaining angle above pi
        else:
            # First unwrap full rotations and then subtract the remaining pi
            wrapped_angle = (angle - (np.pi * wrap_count)) - np.pi
    else:
        # If angle < pi, return the float of it to ensure only float returns
        wrapped_angle = float(angle)

    return wrapped_angle


def wrap_angles_to_pi(angles: list[float]) -> list[float]:
    """Wrap a list of angles between -pi and pi.

    Args:
        angles: The list of angles to be wrapped.

    Returns:
        A list of the wrapped angles.
    """
    # Call wrap_angle_to_pi on given list of angles
    wrapped_angles = list(map(wrap_angle_to_pi, angles))

    return wrapped_angles


def find_new_vertex(link_length: float,
                    vertex1: list[float, float],
                    vertex2: list[float, float]) -> list[float, float]:
    """Find an intermediate vertex along the line from vertex1 -> vertex2 whilst obeying the link length.

    Args:
        link_length: The link length to obey.
        vertex1: The starting vertex of the link.
        vertex2: The ending vertex of the link.

    Returns:
        A list containing the (x,y) cartesian position of the intermediate vertex.
    """
    direction_vector = np.subtract(vertex2, vertex1)  # Pointing from v1 -> v2
    length = np.linalg.norm(direction_vector)
    scaled_direction_vector = (direction_vector / length) * link_length
    adjusted_vertex2_x, adjusted_vertex2_y = np.add(vertex1, scaled_direction_vector)

    return [adjusted_vertex2_x, adjusted_vertex2_y]


def draw_environment(robot_base_radius: float, workspace_width: float = 950.0, workspace_height: float = 950.0) -> None:
    # TODO implement properly
    # Param inconsistencies

    vertices = [(0, -robot_base_radius),
                (0, workspace_height - robot_base_radius),
                (workspace_width, workspace_height - robot_base_radius),
                (workspace_width, -robot_base_radius)]

    plt.plot(*zip(*vertices), 'ko--')
    obstacle = plt.Circle((530, 300), 75, color="black")
    plt.gcf().gca().add_artist(obstacle)


def check_link_lengths(link_lengths: list[float], vertices: dict[str, list[float]]) -> None:
    """Verify the link lengths of the robot after motions.

    Due to imprecision in the floating point calculations, the link lengths might deviate slightly when setting
    vertices. This checks if the link lengths stay within a tolerance of 1e-3.

    Args:
        link_lengths: The desired link lengths.
        vertices: The current vertices.
    """
    # Checking links from first to last
    for i in range(len(vertices["x"]) - 1):
        link_length = link_lengths[i]
        behind_vertex = [vertices["x"][i], vertices["y"][i]]
        ahead_vertex = [vertices["x"][i + 1], vertices["y"][i + 1]]
        dir_vec = np.subtract(ahead_vertex, behind_vertex)
        new_link_length = np.linalg.norm(dir_vec)

        # floating point comparison for link lengths to avoid precision errors
        assert abs(link_length - new_link_length) < 1e-3, f"Link index {i} inconsistent lengths" \
                                                          f" {link_length, new_link_length}"


def validate_target(target: list[float, float], linear_base: bool, arm_reach: float) -> tuple[bool, float]:
    """Validate an IK target for a cartesian space move command.

    Args:
        target: A list containing the (x,y) cartesian space target position.
        linear_base: The bool active state of the linear rail.
        arm_reach: The maximum articulated arm reach.

    Returns:
        A tuple containing a bool to indicate the validity of the target, and a float to show the target's distance.
    """
    valid_target = False

    if linear_base:
        # Linear base allows free movement along x-axis so only y distance matters.
        effective_target_distance = abs(target[1])
    else:
        # Without linear base, the x distance is non-negligible
        effective_target_distance = abs(np.linalg.norm(target))

    # Non-inclusive equality because numerical method IKs such as Fabrik dont do well at the extreme limit
    if effective_target_distance < arm_reach:
        valid_target = True

    return valid_target, effective_target_distance
