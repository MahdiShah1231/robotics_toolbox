import numpy as np


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
    if angle > np.pi:
        wrap_count = angle // (np.pi)
        if wrap_count % 2 == 0:
            wrapped_angle = (angle - (np.pi * wrap_count))
        else:
            wrapped_angle = (angle - (np.pi * wrap_count)) - np.pi
    else:
        wrapped_angle = angle
    return wrapped_angle


def wrap_angles_to_pi(angles):
    wrapped_angles = list(map(wrap_angle_to_pi, angles))
    return wrapped_angles
