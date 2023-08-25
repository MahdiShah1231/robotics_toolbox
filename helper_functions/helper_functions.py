from typing import Union
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum


class MoveType(Enum):
    JOINT = "Joint"  # Joint space command, Forward Kinematics
    CARTESIAN = "Cartesian"  # Cartesian space command, Inverse Kinematics

def calculate_joint_angles(vertices: dict[str, list[float]], linear_base: bool) -> list[float]:
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


def wrap_angle_to_pi(angle: float) -> Union[float, None]:
    if angle is None:
        return angle
    # Wrap angle if angle > pi
    elif angle > np.pi:

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


def wrap_angles_to_pi(angles: Union[list[float], float]) -> Union[list[float], None]:
    if angles is not None:
        # Call wrap_angle_to_pi on given list of angles
        wrapped_angles = list(map(wrap_angle_to_pi, angles))

    # No angles given, return None
    else:
        wrapped_angles = angles

    return wrapped_angles


def find_new_vertex(link_length: float, vertex1: list[float], vertex2: list[float]) -> list[float]:
    direction_vector = np.subtract(vertex2, vertex1)  # Pointing from v1 -> v2
    length = np.linalg.norm(direction_vector)
    scaled_direction_vector = (direction_vector / length) * link_length
    adjusted_vertex2_x, adjusted_vertex2_y = np.add(vertex1, scaled_direction_vector)

    return [adjusted_vertex2_x, adjusted_vertex2_y]


def draw_environment(robot_base_radius: float, workspace_width: float = 950.0, workspace_height: float = 950.0) -> None:
    # Param inconsistencies

    vertices = [(0, -robot_base_radius),
                (0, workspace_height - robot_base_radius),
                (workspace_width, workspace_height - robot_base_radius),
                (workspace_width, -robot_base_radius)]

    plt.plot(*zip(*vertices), 'ko--')
    obstacle = plt.Circle((530, 300), 75, color="black")
    plt.gcf().gca().add_artist(obstacle)


def check_link_lengths(link_lengths: list[float], vertices: dict[str, list[float]]) -> None:

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


def validate_target(target: list[float], linear_base: bool, arm_reach: float) -> tuple[bool, float]:
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
