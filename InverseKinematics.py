import copy
import numpy as np
from helper_functions.helper_functions import calculate_joint_angles, wrap_angle_to_pi, check_link_lengths


class Fabrik:
    def __init__(self, robot,
                 target_position=None,
                 target_orientation=None,
                 error_tolerance=0.000001,
                 max_iterations=100000):
        self.robot = robot
        self.linear_base = robot.linear_base
        self.target_position = list(map(lambda x: x * 1000, target_position))  # Scaling target from m to mm
        self.target_orientation = wrap_angle_to_pi(target_orientation)
        self.error_tolerance = error_tolerance
        self.max_iterations = max_iterations
        self.solved = False

    def solve(self, debug=False, mirror=False):
        iterations = 0
        valid_target = False
        total_robot_length = sum(self.robot.link_lengths)
        if not self.linear_base:
            effective_target_distance = abs(np.linalg.norm(self.target_position))
        else:
            effective_target_distance = abs(self.target_position[1])
        if effective_target_distance < total_robot_length:
            valid_target = True

        if not valid_target:
            print("Could not solve. Target outside of robot range")
            print(f"target distance = {effective_target_distance}, robot max reach = {total_robot_length}")
            print("\nChoose a valid target or link lengths")

        else:
            if self.target_orientation is None: # Target is the last vertex position
                ee_position_actual = [self.robot.vertices["x"][-1],
                                      self.robot.vertices["y"][-1]]
                ee_position_target = self.target_position
                ee_vertex_index = -1

            else: # Target is the second last vertex position
                ee_position_actual = [self.robot.vertices["x"][-2],
                                      self.robot.vertices["y"][-2]]
                last_link_orientation = np.around([np.cos(self.target_orientation), np.sin(self.target_orientation)], decimals = 5)
                oriented_last_link = [element * self.robot.link_lengths[-1] for element in last_link_orientation]
                ee_position_target = np.subtract(self.target_position, oriented_last_link)
                ee_vertex_index = -2
            error_vector = np.subtract(ee_position_target,ee_position_actual)
            error = np.linalg.norm(error_vector)

            while error > self.error_tolerance and iterations < self.max_iterations:
                iterations +=1
                if self.linear_base is False:
                    if self.target_orientation is None:  # Must set n links
                        number_unset_links = self.robot.n_links
                    else:  # Must set n-1 links, last set with correct orientation
                        number_unset_links = self.robot.n_links - 1
                else:
                    if self.target_orientation is None:  # Must set n links
                        number_unset_links = self.robot.n_links - 1
                    else:  # Must set n-2 links, last set with correct orientation and first set with correct orientation
                        number_unset_links = self.robot.n_links - 2
                if iterations %2 != 0:
                    self.robot.vertices["x"][-1] = self.target_position[0]
                    self.robot.vertices["y"][-1] = self.target_position[1]
                    if self.target_orientation is not None:
                        self.robot.vertices["x"][-2] = self.robot.vertices["x"][-1] - oriented_last_link[0]
                        self.robot.vertices["y"][-2] = self.robot.vertices["y"][-1] - oriented_last_link[1]
                    for vertex_number in reversed(range(number_unset_links)):
                        if self.linear_base:
                            vertex_number += 1
                        link_length = self.robot.link_lengths[vertex_number]
                        behind_vertex = [self.robot.vertices["x"][vertex_number + 1],
                                         self.robot.vertices["y"][vertex_number + 1]]
                        ahead_vertex = [self.robot.vertices["x"][vertex_number],
                                        self.robot.vertices["y"][vertex_number]]
                        direction_vector = np.subtract(ahead_vertex, behind_vertex)
                        length = np.linalg.norm(direction_vector)
                        new_direction_vector = (direction_vector * link_length) / length
                        new_vertex_x, new_vertex_y = np.add(behind_vertex, new_direction_vector)
                        self.robot.vertices["x"][vertex_number] = new_vertex_x
                        self.robot.vertices["y"][vertex_number] = new_vertex_y

                    if self.linear_base:
                        error_vector = np.subtract([self.robot.vertices["x"][1],
                                                    self.robot.vertices["y"][1]], [self.robot.vertices["x"][1], 0])
                    else:
                        error_vector = np.subtract([self.robot.vertices["x"][0],
                                                    self.robot.vertices["y"][0]], self.robot.robot_base_origin)
                    error = np.linalg.norm(error_vector)
                    if self.linear_base:
                        base_offset_vec = np.subtract([self.robot.vertices["x"][1],
                                                       self.robot.vertices["y"][1]], self.robot.robot_base_origin)
                        base_offset = base_offset_vec[0]
                        self.robot.link_lengths[0] = abs(base_offset)
                else:
                    self.robot.vertices["x"][0] = self.robot.robot_base_origin[0]
                    self.robot.vertices["y"][0] = self.robot.robot_base_origin[1]
                    if self.linear_base:
                        self.robot.vertices["y"][1] = self.robot.vertices["y"][0]
                    for vertex_number in range(1, number_unset_links+1):
                        if self.linear_base:
                            vertex_number += 1
                        link_length = self.robot.link_lengths[vertex_number - 1]
                        behind_vertex = [self.robot.vertices["x"][vertex_number - 1],
                                         self.robot.vertices["y"][vertex_number - 1]]
                        ahead_vertex = [self.robot.vertices["x"][vertex_number],
                                        self.robot.vertices["y"][vertex_number]]
                        direction_vector = np.subtract(ahead_vertex, behind_vertex)
                        length = np.linalg.norm(direction_vector)
                        new_direction_vector = (direction_vector * link_length) / length
                        new_vertex_x, new_vertex_y = np.add(behind_vertex, new_direction_vector)
                        self.robot.vertices["x"][vertex_number] = new_vertex_x
                        self.robot.vertices["y"][vertex_number] = new_vertex_y
                    if self.target_orientation is not None:
                        self.robot.vertices["x"][-1] = self.robot.vertices["x"][-2] + oriented_last_link[0]
                        self.robot.vertices["y"][-1] = self.robot.vertices["y"][-2] + oriented_last_link[1]

                    ee_position_actual = [self.robot.vertices["x"][ee_vertex_index],
                                          self.robot.vertices["y"][ee_vertex_index]]
                    error_vector = np.subtract(ee_position_target,ee_position_actual)
                    error = np.linalg.norm(error_vector)
            if error > self.error_tolerance:
                print("Could not solve.")
                print(f"error: {error}\n")
                print("Computed link lengths: ")
                check_link_lengths(link_lengths=self.robot.link_lengths, vertices=self.robot.vertices)
                print("\nVertices: ")
                print(self.robot.vertices)

            else:
                print("Final robot configuration:")
                print(self.robot.vertices)
                if mirror:
                    self.__mirrored_elbows()
                    print(self.robot.mirrored_vertices)
                print("Final joint angles:")
                self.robot.joint_configuration = calculate_joint_angles(self.robot.vertices)
                print(self.robot.joint_configuration)
                if mirror:
                    self.robot.mirrored_joint_configuration = calculate_joint_angles(self.robot.mirrored_vertices)
                    print(self.robot.mirrored_joint_configuration)

                if debug:
                    print("\nPrinting lengths for debugging...")
                    print("\nRobot link lengths:")
                    check_link_lengths(link_lengths=self.robot.link_lengths, vertices=self.robot.vertices)
                    if mirror:
                        print("\nRobot link lengths (mirrored vertices):")
                        check_link_lengths(link_lengths=self.robot.link_lengths, vertices=self.robot.mirrored_vertices)
                self.solved = True
        return self.robot.vertices

    def __mirrored_elbows(self):
        self.robot.mirrored_vertices = copy.deepcopy(self.robot.vertices)
        if self.linear_base:
            start = [self.robot.mirrored_vertices["x"][1], self.robot.mirrored_vertices["y"][1]]
        else:
            start = [self.robot.mirrored_vertices["x"][0], self.robot.mirrored_vertices["y"][0]]
        if self.target_orientation is None:
            last_vertex_index = len(self.robot.mirrored_vertices["x"])-1
        else:
            last_vertex_index = len(self.robot.mirrored_vertices["x"])-2
        last_vertex = [self.robot.mirrored_vertices["x"][last_vertex_index],
                       self.robot.mirrored_vertices["y"][last_vertex_index]]
        mirror_vec = np.subtract(last_vertex, start)
        mirror_vec_length = np.linalg.norm(mirror_vec)
        if self.linear_base:
            start_vertex_index = 2
        else:
            start_vertex_index = 1

        for vertex_index in range(start_vertex_index, last_vertex_index):
            vertex = [self.robot.mirrored_vertices["x"][vertex_index],
                      self.robot.mirrored_vertices["y"][vertex_index]]
            link_length = self.robot.link_lengths[vertex_index-1]
            direction = np.subtract(vertex,start)
            direction_cosine = np.dot(mirror_vec,direction)/(link_length*mirror_vec_length)
            scaled_mirror_vec = (link_length*direction_cosine*mirror_vec)/mirror_vec_length
            midpoint = np.add(start, scaled_mirror_vec)
            translation_direction = np.subtract(midpoint,vertex)
            scaled_translation_direction = translation_direction*2
            new_vertex = np.add(vertex,scaled_translation_direction)
            self.robot.mirrored_vertices["x"][vertex_index] = new_vertex[0]
            self.robot.mirrored_vertices["y"][vertex_index] = new_vertex[1]

        return self.robot.mirrored_vertices

    def check_collisions(self):
        robot_points = list(zip(self.robot.vertices["x"], self.robot.vertices["y"]))
        # print(robot_points)
        # collision_checking('Scenario_01_OT2_IntelliX_001.ppm', self.configuration.vertices, check_radius=15, granularity=10)

if __name__ == '__main__':
    pass

# "Scenario_01_OT2_IntelliX_001.ppm"