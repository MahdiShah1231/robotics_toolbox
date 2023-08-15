import copy
import numpy as np
from helper_functions.helper_functions import calculate_joint_angles, wrap_angle_to_pi, check_link_lengths, \
    validate_target, find_new_vertex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot import Robot

SCALE_TO_MM = 1000


class Fabrik:
    def __init__(self, robot: 'Robot',
                 target_position: list[float],
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

    def solve(self, debug: bool, mirror: bool, move: bool) -> tuple[list[float], list[float]]:
        iterations = 0
        joint_configuration = self.__robot.joint_configuration
        mirrored_joint_configuration = None

        # First validate target before attempting to compute solution
        valid_target, effective_target_distance = validate_target(target=self.__target_position,
                                                                  linear_base=self.__robot.linear_base,
                                                                  robot_length=self.__total_robot_length)
        # Invalid target, no computation attempted.
        if not valid_target:
            print("Could not solve. Target outside of robot range")
            print(f"target distance = {effective_target_distance}, robot max reach = {self.__total_robot_length}")
            print("\nChoose a valid target or link lengths")

        # Valid target given, attempt to compute IK solution.
        else:
            # If there is no target orientation given, the end effector can be oriented in any way to reach the target
            if self.__target_orientation is None:
                # The "effective" end effector = robot last vertex
                ee_position_actual = [self.__robot.vertices["x"][-1],
                                      self.__robot.vertices["y"][-1]]
                # The target for the effective end effector is = target
                ee_position_target = self.__target_position
                ee_vertex_index = -1  # Last vertex is the effective ee

            # If there is a target orientation given, the end effector must be oriented in the specified orientation
            else:
                # The "effective" end effector = robot 2nd last vertex, This is the start of the last link,
                # it must be moved to an "effective" target such that with the correct orientation, the last vertex will
                # reach the target
                ee_position_actual = [self.__robot.vertices["x"][-2],
                                      self.__robot.vertices["y"][-2]]
                last_link_orientation = np.around([np.cos(self.__target_orientation),
                                                   np.sin(self.__target_orientation)], decimals=5)
                oriented_last_link = list(map(lambda i: i * self.__robot.link_lengths[-1], last_link_orientation))
                # The "effective" target has to take into account the orientation of the final link
                ee_position_target = np.subtract(self.__target_position, oriented_last_link)
                ee_vertex_index = -2  # Second last vertex is the effective ee

            if not self.__robot.linear_base:
                n_arm_links = self.__robot.n_links
                arm_start_link_idx = 0  # Arm starts from first link if no linear base
            else:
                n_arm_links = self.__robot.n_links - 1
                arm_start_link_idx = 1  # Arm starts from second link if no linear base

            # n_unset_arm_links = number of links to set through computation.
            # Equivalent to n_arm_links if no target orientation (e.g need to set all arm links)
            # Or equals n_arm_links - 1 if target orientation given (e.g no need to computationally set last link)
            n_unset_arm_links = n_arm_links if self.__target_orientation is None else n_arm_links - 1

            # last unset arm link idx takes into account the extra starting link if linear base is present
            last_unset_arm_link_idx = n_unset_arm_links - 1 if not self.__robot.linear_base else n_unset_arm_links

            # Calculate starting error and then start computation
            error_vector = np.subtract(ee_position_target, ee_position_actual)
            error = np.linalg.norm(error_vector)

            vertices = copy.deepcopy(self.__robot.vertices)
            link_lengths = copy.deepcopy(self.__robot.link_lengths)

            while error > self.__error_tolerance and iterations < self.__max_iterations:
                iterations += 1
                if iterations % 2 != 0:  # Odd iteration = backward iteration
                    # First step of backwards iteration is to move last vertex to the target
                    vertices["x"][-1] = self.__target_position[0]
                    vertices["y"][-1] = self.__target_position[1]

                    if self.__target_orientation is not None:
                        # Setting second last vertex to give the correct target orientation of last link
                        vertices["x"][-2] = vertices["x"][-1] - oriented_last_link[0]
                        vertices["y"][-2] = vertices["y"][-1] - oriented_last_link[1]

                    # Only setting the arm links
                    # Vertex i is the start of link i
                    # When iterating backwards, the ahead vertex is the start of the current link, and back vertex
                    # is the end of the current link.
                    for vertex_index in reversed(range(arm_start_link_idx, last_unset_arm_link_idx + 1)):
                        link_length = link_lengths[vertex_index]

                        behind_vertex = [vertices["x"][vertex_index + 1],
                                         vertices["y"][vertex_index + 1]]
                        ahead_vertex = [vertices["x"][vertex_index],
                                        vertices["y"][vertex_index]]

                        # Adjust the ahead vertex based on the link length and direction
                        new_ahead_vertex = find_new_vertex(link_length=link_length,
                                                           vertex1=behind_vertex,
                                                           vertex2=ahead_vertex)

                        vertices["x"][vertex_index] = new_ahead_vertex[0]
                        vertices["y"][vertex_index] = new_ahead_vertex[1]

                    if self.__robot.linear_base:
                        # Base position is the end of the prismatic link
                        # Base position target arbitrarily anywhere on x axis if linear base enabled
                        robot_base_pos_current = [vertices["x"][1], vertices["y"][1]]
                        robot_base_pos_target = [vertices["x"][1], 0]

                        # Base offset vec is the distance the prismatic joint needs to extend to go from
                        # origin -> current pos
                        base_offset_vec = np.subtract(robot_base_pos_current, self.__robot.robot_base_origin)
                        base_offset = base_offset_vec[0]  # Only care about x distance
                        link_lengths[0] = abs(base_offset)  # Updating prismatic link length

                    else:
                        # Base position is start of the robot arm
                        # Base position target at the robot base origin
                        robot_base_pos_current = [vertices["x"][0], vertices["y"][0]]
                        robot_base_pos_target = self.__robot.robot_base_origin

                    error_vector = np.subtract(robot_base_pos_current, robot_base_pos_target)
                    error = np.linalg.norm(error_vector)

                else:  # Even iteration = forward iteration
                    # First step of forward iteration is to reset the robot base position
                    # TODO check if this is needed for all cases or only if no linear base
                    vertices["x"][0] = self.__robot.robot_base_origin[0]
                    vertices["y"][0] = self.__robot.robot_base_origin[1]
                    if self.__robot.linear_base:
                        vertices["y"][1] = self.__robot.robot_base_origin[1]
                        vertices["x"][1] = robot_base_pos_current[0]

                    for vertex_index in range(arm_start_link_idx, last_unset_arm_link_idx + 1):
                        link_length = link_lengths[vertex_index]

                        behind_vertex = [vertices["x"][vertex_index],
                                         vertices["y"][vertex_index]]
                        ahead_vertex = [vertices["x"][vertex_index + 1],
                                        vertices["y"][vertex_index + 1]]

                        # Adjust the ahead vertex based on the link length and direction
                        new_ahead_vertex = find_new_vertex(link_length=link_length,
                                                           vertex1=behind_vertex,
                                                           vertex2=ahead_vertex)

                        vertices["x"][vertex_index + 1] = new_ahead_vertex[0]
                        vertices["y"][vertex_index + 1] = new_ahead_vertex[1]

                    if self.__target_orientation is not None:
                        # Resetting last arm vertex to give correct orientation from the second last vertex
                        # Second last vertex is the last vertex set in the forward iteration if a targ orientation given
                        vertices["x"][-1] = vertices["x"][-2] + oriented_last_link[0]
                        vertices["y"][-1] = vertices["y"][-2] + oriented_last_link[1]

                    ee_position_actual = [vertices["x"][ee_vertex_index],
                                          vertices["y"][ee_vertex_index]]
                    error_vector = np.subtract(ee_position_target, ee_position_actual)
                    error = np.linalg.norm(error_vector)

            if error > self.__error_tolerance:  # No solution found, maxed out iterations
                print("Could not solve.")
                print(f"error: {error}\n")
                if debug:
                    check_link_lengths(link_lengths=link_lengths, vertices=vertices)
                    print("\nVertices: ")
                    print(vertices)

            else:  # Solution found
                # Calculate joint angles for the solution
                joint_configuration = calculate_joint_angles(vertices=vertices, linear_base=self.__robot.linear_base)
                if self.__robot.linear_base:
                    joint_configuration[0] = link_lengths[0] * np.sign(vertices["x"][1])

                if mirror:
                    # Find the vertices and joint angles for the mirrored solution
                    mirrored_vertices = self._get_mirror_configuration(vertices)
                    mirrored_joint_configuration = calculate_joint_angles(vertices=mirrored_vertices, linear_base=self.__robot.linear_base)
                    mirrored_joint_configuration[0] = joint_configuration[0]

                self.solved = True

                if move:
                    self.__robot.vertices = vertices
                    self.__robot.joint_configuration = joint_configuration
                    self.__robot.link_lengths = link_lengths
                    if mirror:
                        self.__robot.mirrored_vertices = mirrored_vertices
                        self.__robot.mirrored_joint_configuration = mirrored_joint_configuration

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
                    check_link_lengths(link_lengths=self.__robot.link_lengths, vertices=self.__robot.vertices)
                    if mirror:
                        check_link_lengths(link_lengths=self.__robot.link_lengths,
                                           vertices=self.__robot.mirrored_vertices)
                    print(f"\nSolution found in {iterations} iterations")

        return joint_configuration, mirrored_joint_configuration

    def _get_mirror_configuration(self, vertices) -> dict[str, list[float]]:
        mirrored_vertices = copy.deepcopy(vertices)  # Start by copying current vertices

        # The first robot arm vertex is where the mirror line starts
        mirror_line_start_vertex_index = 0 if not self.__robot.linear_base else 1
        mirror_line_start_vertex = [mirrored_vertices["x"][mirror_line_start_vertex_index],
                                    mirrored_vertices["y"][mirror_line_start_vertex_index]]

        # The mirror line ends at the last vertex if theres no orientation, or second last if there is an orientation
        # n_links updates based on if a linear base exists.
        # vertices[n_links] = last vertex, vertices[n_links - 1] = second last vertex
        mirror_line_last_vertex_index = self.__robot.n_links if self.__target_orientation is None else self.__robot.n_links - 1
        mirror_line_last_vertex = [mirrored_vertices["x"][mirror_line_last_vertex_index],
                                   mirrored_vertices["y"][mirror_line_last_vertex_index]]

        mirror_vec = np.subtract(mirror_line_last_vertex, mirror_line_start_vertex)
        mirror_vec_length = np.linalg.norm(mirror_vec)
        unit_mirror_vec = mirror_vec/mirror_vec_length

        # Iterate through points not including the start and end as they are on the mirror line so don't get transformed
        # Mirroring vertices using the vector line formed between the mirror start vertex and the vertex
        for vertex_index in range(mirror_line_start_vertex_index + 1, mirror_line_last_vertex_index):
            vertex = [mirrored_vertices["x"][vertex_index],
                      mirrored_vertices["y"][vertex_index]]

            direction_vector = np.subtract(vertex, mirror_line_start_vertex)
            length = np.linalg.norm(direction_vector)
            unit_dir_vec = direction_vector/length
            direction_cosine = np.dot(unit_dir_vec, unit_mirror_vec)  # cos(theta) between mirror line and vector line
            line_projection = length * direction_cosine * unit_mirror_vec  # projecting the direction vector on mirror line
            midpoint = np.add(mirror_line_start_vertex, line_projection)  # midpoint along line from vertex -> mirrored vertex
            translation_direction = np.subtract(midpoint, vertex)  # direction vector pointing vertex -> mirrored vertex
            mirrored_vertex = vertex + (2 * translation_direction)  # Using symmetry (vertex -> midpoint = midpoint -> mirrored verted)

            mirrored_vertices["x"][vertex_index] = mirrored_vertex[0]
            mirrored_vertices["y"][vertex_index] = mirrored_vertex[1]

        return mirrored_vertices

    def check_collisions(self):
        raise NotImplementedError

if __name__ == '__main__':
    pass
