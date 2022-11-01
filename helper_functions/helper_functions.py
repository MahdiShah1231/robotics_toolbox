import numpy as np
from matplotlib import pyplot as plt


def calculate_joint_angles(vertices):
    joint_angles = [0] * (len(vertices["x"]) - 1)
    old_direction_vector = [1, 0]
    last_vertex_index = len(vertices["x"]) - 1
    for vertex_index in range(last_vertex_index):
        vertex = [vertices["x"][vertex_index], vertices["y"][vertex_index]]
        vertex_ahead = [vertices["x"][vertex_index + 1], vertices["y"][vertex_index + 1]]
        new_direction_vector = np.subtract(vertex_ahead, vertex)
        arg1 = old_direction_vector[0] * new_direction_vector[1] - old_direction_vector[1] * new_direction_vector[0]
        arg2 = old_direction_vector[0] * new_direction_vector[0] + old_direction_vector[1] * new_direction_vector[1]
        joint_angle = np.arctan2(arg1, arg2)
        joint_angles[vertex_index] = joint_angle
        old_direction_vector = new_direction_vector
    return joint_angles


def wrap_angle_to_pi(angle):
    if angle is not None:
        if angle > np.pi:
            wrap_count = angle // (np.pi)
            if wrap_count % 2 == 0:
                wrapped_angle = (angle - (np.pi * wrap_count))
            else:
                wrapped_angle = (angle - (np.pi * wrap_count)) - np.pi
        else:
            wrapped_angle = angle
    else:
        wrapped_angle = angle
    return wrapped_angle


def wrap_angles_to_pi(angles):
    if angles is not None:
        wrapped_angles = list(map(wrap_angle_to_pi, angles))
    else:
        wrapped_angles = None
    return wrapped_angles


def draw_environment(robot_base_radius, workspace_width=None, workspace_height=None):
    # GIVE PARAMS IN MM
    if workspace_width is None:
        workspace_width = 950
    if workspace_height is None:
        workspace_height = 950

    vertices = [(0, -robot_base_radius),
                (0, workspace_height - robot_base_radius),
                (workspace_width, workspace_height - robot_base_radius),
                (workspace_width, -robot_base_radius)]

    plt.plot(*zip(*vertices), 'ko--')
    obstacle = plt.Circle((530, 300), 75, color="black")
    plt.gcf().gca().add_artist(obstacle)


def check_link_lengths(link_lengths, vertices):
    for i in range(len(vertices["x"]) - 1):  # Checking always forward
        link_length = link_lengths[i]
        behind_vertex = [vertices["x"][i], vertices["y"][i]]
        ahead_vertex = [vertices["x"][i + 1], vertices["y"][i + 1]]
        dir_vec = np.subtract(ahead_vertex, behind_vertex)
        new_link_length = np.linalg.norm(dir_vec)
        print(link_length, new_link_length)

def validate_target(target, linear_base, robot_length):
    valid_target = False
    if linear_base:  # Linear base allows free movement along x-axis so only y distance matters.
        effective_target_distance = abs(target[1])
    else:
        effective_target_distance = abs(np.linalg.norm(target))

    if effective_target_distance < robot_length:
        valid_target = True

    return valid_target, effective_target_distance
