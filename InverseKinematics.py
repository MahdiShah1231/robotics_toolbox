import copy
import numpy as np
from helper_functions.helper_functions import calculate_joint_angles, wrap_angle_to_pi, check_link_lengths, \
    validate_target
from typing import TYPE_CHECKING, List, Dict

if TYPE_CHECKING:
    from Robot import Robot

SCALE_TO_MM = 1000

class Fabrik:
    def __init__(self, robot: 'Robot',
                 target_position: List[float],
                 target_orientation: float,
                 error_tolerance: float = 0.000001,
                 max_iterations: int = 100000) -> None:
        self.__robot = robot
        self.__target_position = list(map(lambda x: x * SCALE_TO_MM, target_position))  # Scaling target from m to mm
        self.__target_orientation = wrap_angle_to_pi(target_orientation)
        self.__effective_target_distance = None
        self.__total_robot_length = sum(self.__robot.link_lengths)
        self.__error_tolerance = error_tolerance
        self.__max_iterations = max_iterations
        self.solved = False

    def solve(self, debug: bool, mirror: bool) -> Dict[str, List[float]]:
        iterations = 0
        valid_target, effective_target_distance = validate_target(target=self.__target_position,
                                                                  linear_base=self.__robot.linear_base,
                                                                  robot_length=self.__total_robot_length)
        if not valid_target:
            print("Could not solve. Target outside of robot range")
            print(f"target distance = {effective_target_distance}, robot max reach = {self.__total_robot_length}")
            print("\nChoose a valid target or link lengths")

        else:
            if self.__target_orientation is None:  # Target is the last vertex position
                ee_position_actual = [self.__robot.vertices["x"][-1],
                                      self.__robot.vertices["y"][-1]]
                ee_position_target = self.__target_position
                ee_vertex_index = -1

            else:  # Target is the second last vertex position
                ee_position_actual = [self.__robot.vertices["x"][-2],
                                      self.__robot.vertices["y"][-2]]
                last_link_orientation = np.around([np.cos(self.__target_orientation),
                                                   np.sin(self.__target_orientation)], decimals=5)
                oriented_last_link = list(map(lambda i: i * self.__robot.link_lengths[-1], last_link_orientation))
                ee_position_target = np.subtract(self.__target_position, oriented_last_link)
                ee_vertex_index = -2

            error_vector = np.subtract(ee_position_target, ee_position_actual)
            error = np.linalg.norm(error_vector)

            if not self.__robot.linear_base:
                start_link_idx = 0
                if self.__target_orientation is None:  # Must set n links
                    n_unset_links = self.__robot.n_links
                else:
                    # Must set n-1 links, last link already set with correct orientation
                    n_unset_links = self.__robot.n_links - 1
                last_link_idx = n_unset_links
            else:
                start_link_idx = 1
                if self.__target_orientation is None:  # Must set n - 1 links
                    n_unset_links = self.__robot.n_links - 1
                else:
                    # Must set n-2 links, last set with correct orientation and
                    # first set with correct orientation (prismatic base)
                    n_unset_links = self.__robot.n_links - 2
                last_link_idx = n_unset_links + 1

            while error > self.__error_tolerance and iterations < self.__max_iterations:
                iterations += 1
                if iterations % 2 != 0:  # Odd iteration = backward iteration
                    self.__robot.vertices["x"][-1] = self.__target_position[0]
                    self.__robot.vertices["y"][-1] = self.__target_position[1]

                    if self.__target_orientation is not None:
                        self.__robot.vertices["x"][-2] = self.__robot.vertices["x"][-1] - oriented_last_link[0]
                        self.__robot.vertices["y"][-2] = self.__robot.vertices["y"][-1] - oriented_last_link[1]

                    for vertex_index in reversed(range(start_link_idx, last_link_idx)):
                        link_length = self.__robot.link_lengths[vertex_index]
                        behind_vertex = [self.__robot.vertices["x"][vertex_index + 1],
                                         self.__robot.vertices["y"][vertex_index + 1]]
                        ahead_vertex = [self.__robot.vertices["x"][vertex_index],
                                        self.__robot.vertices["y"][vertex_index]]
                        direction_vector = np.subtract(ahead_vertex, behind_vertex)
                        length = np.linalg.norm(direction_vector)
                        new_direction_vector = (direction_vector / length) * link_length
                        new_vertex_x, new_vertex_y = np.add(behind_vertex, new_direction_vector)
                        self.__robot.vertices["x"][vertex_index] = new_vertex_x
                        self.__robot.vertices["y"][vertex_index] = new_vertex_y

                    if self.__robot.linear_base:
                        error_vector = np.subtract([self.__robot.vertices["x"][1], self.__robot.vertices["y"][1]],
                                                   [self.__robot.vertices["x"][1], 0])
                        base_offset_vec = np.subtract([self.__robot.vertices["x"][1], self.__robot.vertices["y"][1]],
                                                      self.__robot.robot_base_origin)
                        base_offset = base_offset_vec[0]
                        self.__robot.link_lengths[0] = abs(base_offset)  # Correcting error in x with linear base
                    else:
                        error_vector = np.subtract([self.__robot.vertices["x"][0],
                                                    self.__robot.vertices["y"][0]], self.__robot.robot_base_origin)
                    error = np.linalg.norm(error_vector)

                else:  # Even iteration = forward iteration
                    self.__robot.vertices["x"][0] = self.__robot.robot_base_origin[0]
                    self.__robot.vertices["y"][0] = self.__robot.robot_base_origin[1]
                    if self.__robot.linear_base:
                        self.__robot.vertices["y"][1] = self.__robot.robot_base_origin[1]

                    for vertex_index in range(start_link_idx + 1, last_link_idx + 1):
                        link_length = self.__robot.link_lengths[vertex_index - 1]
                        behind_vertex = [self.__robot.vertices["x"][vertex_index - 1],
                                         self.__robot.vertices["y"][vertex_index - 1]]
                        ahead_vertex = [self.__robot.vertices["x"][vertex_index],
                                        self.__robot.vertices["y"][vertex_index]]
                        direction_vector = np.subtract(ahead_vertex, behind_vertex)
                        length = np.linalg.norm(direction_vector)
                        new_direction_vector = (direction_vector / length) * link_length
                        new_vertex_x, new_vertex_y = np.add(behind_vertex, new_direction_vector)
                        self.__robot.vertices["x"][vertex_index] = new_vertex_x
                        self.__robot.vertices["y"][vertex_index] = new_vertex_y

                    if self.__target_orientation is not None:
                        self.__robot.vertices["x"][-1] = self.__robot.vertices["x"][-2] + oriented_last_link[0]
                        self.__robot.vertices["y"][-1] = self.__robot.vertices["y"][-2] + oriented_last_link[1]

                    ee_position_actual = [self.__robot.vertices["x"][ee_vertex_index],
                                          self.__robot.vertices["y"][ee_vertex_index]]
                    error_vector = np.subtract(ee_position_target, ee_position_actual)
                    error = np.linalg.norm(error_vector)

            if error > self.__error_tolerance:
                print("Could not solve.")
                print(f"error: {error}\n")
                if debug:
                    print("Computed link lengths: ")
                    check_link_lengths(link_lengths=self.__robot.link_lengths, vertices=self.__robot.vertices)
                    print("\nVertices: ")
                    print(self.__robot.vertices)

            else:
                if mirror:
                    self.__mirrored_elbows()
                    self.__robot.mirrored_joint_configuration = calculate_joint_angles(self.__robot.mirrored_vertices)
                self.__robot.joint_configuration = calculate_joint_angles(self.__robot.vertices)
                self.solved = True

                if debug:
                    print("\nPrinting debugging info...")
                    print("Final robot configuration:")
                    print(self.__robot.vertices)
                    if mirror:
                        print(self.__robot.mirrored_vertices)
                    print("Final joint angles:")
                    print(self.__robot.joint_configuration)
                    if mirror:
                        print(self.__robot.mirrored_joint_configuration)
                    print("\nRobot link lengths:")
                    check_link_lengths(link_lengths=self.__robot.link_lengths, vertices=self.__robot.vertices)
                    if mirror:
                        print("\nRobot link lengths (mirrored vertices):")
                        check_link_lengths(link_lengths=self.__robot.link_lengths, vertices=self.__robot.mirrored_vertices)
                    print(f"\nSolution found in {iterations} iterations")

        return self.__robot.vertices

    def __mirrored_elbows(self) -> Dict[str, List[float]]:
        self.__robot.mirrored_vertices = copy.deepcopy(self.__robot.vertices)
        if self.__robot.linear_base:
            mirror_start_vertex_index = 1
        else:
            mirror_start_vertex_index = 0
        start = [self.__robot.mirrored_vertices["x"][mirror_start_vertex_index], self.__robot.mirrored_vertices["y"][mirror_start_vertex_index]]

        if self.__target_orientation is None:
            mirror_last_vertex_index = self.__robot.n_links
        else:
            mirror_last_vertex_index = self.__robot.n_links - 1

        last_vertex = [self.__robot.mirrored_vertices["x"][mirror_last_vertex_index],
                       self.__robot.mirrored_vertices["y"][mirror_last_vertex_index]]
        mirror_vec = np.subtract(last_vertex, start)
        mirror_vec_length = np.linalg.norm(mirror_vec)

        if self.__robot.linear_base:
            start_vertex_index = 2
        else:
            start_vertex_index = 1

        for vertex_index in range(start_vertex_index, mirror_last_vertex_index):
            vertex = [self.__robot.mirrored_vertices["x"][vertex_index],
                      self.__robot.mirrored_vertices["y"][vertex_index]]
            direction_vector = np.subtract(vertex, start)
            length = np.linalg.norm(direction_vector)
            direction_cosine = np.dot(direction_vector, mirror_vec)/(length*mirror_vec_length) # cos_theta = (A.B)/(|A|*|B|)
            scaled_mirror_vec = (length * direction_cosine) * (mirror_vec / mirror_vec_length)
            midpoint = np.add(start, scaled_mirror_vec)
            translation_direction = np.subtract(midpoint, vertex)
            scaled_translation_direction = translation_direction*2
            new_vertex = np.add(vertex, scaled_translation_direction)
            self.__robot.mirrored_vertices["x"][vertex_index] = new_vertex[0]
            self.__robot.mirrored_vertices["y"][vertex_index] = new_vertex[1]

        return self.__robot.mirrored_vertices

    def check_collisions(self):
        raise NotImplementedError

if __name__ == '__main__':
    pass
