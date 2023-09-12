import copy
import logging
from abc import ABC, abstractmethod

import numpy as np

from helper_functions.helper_functions import calculate_joint_angles, wrap_angle_to_pi, validate_target, \
    find_new_vertex, create_logger

SCALE_TO_MM = 1000
logger = create_logger(module_name=__name__, level=logging.INFO)  # Change debug level as needed


class IKSolverBase(ABC):
    def __init__(self,
                 error_tolerance: float,
                 max_iterations: int) -> None:
        self.solved = False
        self._error_tolerance = error_tolerance
        self._max_iterations = max_iterations
        self._target_position = None
        self._target_orientation = None

    def setup_target(self,
                     target_position: list[float, float],
                     target_orientation: float) -> None:
        self.solved = False
        self._target_position = list(map(lambda x: x * SCALE_TO_MM, target_position))  # Scaling target from m to mm
        self._target_orientation = wrap_angle_to_pi(target_orientation)

    @abstractmethod
    def solve(self,
              vertices: dict[str, list[float]],
              link_lengths: list[float],
              linear_base: bool,
              robot_base_origin: list[float],
              start_config: list[float],
              mirror: bool) -> dict:

        raise NotImplementedError


class FabrikSolver(IKSolverBase):
    def __init__(self,
                 error_tolerance: float = 0.000001,
                 max_iterations: int = 100000) -> None:
        super().__init__(error_tolerance=error_tolerance,
                         max_iterations=max_iterations)

    def solve(self,
              vertices: dict[str, list[float]],
              link_lengths: list[float],
              linear_base: bool,
              robot_base_origin: list[float],
              start_config: dict[str, list[float]],
              mirror: bool) -> dict:

        if mirror:
            logger.warning("Mirror functionality doesn't currently work")

        # Create deep copy of vertices to prevent modifying robot vertices directly.
        vertices = copy.deepcopy(vertices)

        iterations = 0
        mirrored_joint_configuration = copy.deepcopy(start_config)
        mirrored_vertices = vertices
        solution = {
            "joint_config": start_config,
            "mirrored_joint_config": mirrored_joint_configuration,
            "vertices": vertices,
            "mirrored_vertices": mirrored_vertices,
            "link_lengths": link_lengths
        }

        n_links = len(link_lengths)
        if not linear_base:
            max_reach = sum(link_lengths)
        else:
            max_reach = sum(link_lengths[1::])

        # First validate target before attempting to compute solution
        valid_target, effective_target_distance = validate_target(target=self._target_position,
                                                                  linear_base=linear_base,
                                                                  arm_reach=max_reach)
        # Invalid target, no computation attempted.
        if not valid_target:
            logger.warning("Could not solve IK. Target outside of robot range.")
            logger.warning(f"Target distance = {effective_target_distance}, robot max reach = {max_reach}.")
            logger.warning("Choose a valid target or link lengths.")

        # Valid target given, attempt to compute IK solution.
        else:
            # If there is no target orientation given, the end effector can be oriented in any way to reach the target
            if self._target_orientation is None:
                # The "effective" end effector = robot last vertex
                ee_position_actual = [vertices["x"][-1], vertices["y"][-1]]
                # The target for the effective end effector is = target
                ee_position_target = self._target_position
                ee_vertex_index = -1  # Last vertex is the effective ee

            # If there is a target orientation given, the end effector must be oriented in the specified orientation
            else:
                # The "effective" end effector = robot 2nd last vertex, This is the start of the last link,
                # it must be moved to an "effective" target such that with the correct orientation, the last vertex will
                # reach the target
                ee_position_actual = [vertices["x"][-2], vertices["y"][-2]]
                last_link_orientation = np.around([np.cos(self._target_orientation),
                                                   np.sin(self._target_orientation)], decimals=5)
                oriented_last_link = list(map(lambda i: i * link_lengths[-1], last_link_orientation))
                # The "effective" target has to take into account the orientation of the final link
                ee_position_target = np.subtract(self._target_position, oriented_last_link)
                ee_vertex_index = -2  # Second last vertex is the effective ee

            if not linear_base:
                n_arm_links = n_links
                arm_start_link_idx = 0  # Arm starts from first link if no linear base
            else:
                n_arm_links = n_links - 1
                arm_start_link_idx = 1  # Arm starts from second link if no linear base

            # n_unset_arm_links = number of links to set through computation.
            # Equivalent to n_arm_links if no target orientation (e.g need to set all arm links)
            # Or equals n_arm_links - 1 if target orientation given (e.g no need to computationally set last link)
            n_unset_arm_links = n_arm_links if self._target_orientation is None else n_arm_links - 1

            # last unset arm link idx takes into account the extra starting link if linear base is present
            last_unset_arm_link_idx = n_unset_arm_links - 1 if not linear_base else n_unset_arm_links

            # Calculate starting error and then start computation
            error_vector = np.subtract(ee_position_target, ee_position_actual)
            error = np.linalg.norm(error_vector)

            while error > self._error_tolerance and iterations < self._max_iterations:
                iterations += 1
                if iterations % 2 != 0:  # Odd iteration = backward iteration
                    # First step of backwards iteration is to move last vertex to the target
                    vertices["x"][-1] = self._target_position[0]
                    vertices["y"][-1] = self._target_position[1]

                    if self._target_orientation is not None:
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

                    if linear_base:
                        # Base position is the end of the prismatic link
                        # Base position target arbitrarily anywhere on x axis if linear base enabled
                        robot_base_pos_current = [vertices["x"][1], vertices["y"][1]]
                        robot_base_pos_target = [vertices["x"][1], 0]

                        # Base offset vec is the distance the prismatic joint needs to extend to go from
                        # origin -> current pos
                        base_offset_vec = np.subtract(robot_base_pos_current, robot_base_origin)
                        base_offset = base_offset_vec[0]  # Only care about x distance
                        link_lengths[0] = abs(base_offset)  # Updating prismatic link length

                    else:
                        # Base position is start of the robot arm
                        # Base position target at the robot base origin
                        robot_base_pos_current = [vertices["x"][0], vertices["y"][0]]
                        robot_base_pos_target = robot_base_origin

                    error_vector = np.subtract(robot_base_pos_current, robot_base_pos_target)
                    error = np.linalg.norm(error_vector)

                else:  # Even iteration = forward iteration
                    # First step of forward iteration is to reset the robot base position
                    # TODO check if this is needed for all cases or only if no linear base
                    vertices["x"][0] = robot_base_origin[0]
                    vertices["y"][0] = robot_base_origin[1]
                    if linear_base:
                        vertices["y"][1] = robot_base_origin[1]
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

                    if self._target_orientation is not None:
                        # Resetting last arm vertex to give correct orientation from the second last vertex
                        # Second last vertex is the last vertex set in the forward iteration if a targ orientation given
                        vertices["x"][-1] = vertices["x"][-2] + oriented_last_link[0]
                        vertices["y"][-1] = vertices["y"][-2] + oriented_last_link[1]

                    ee_position_actual = [vertices["x"][ee_vertex_index],
                                          vertices["y"][ee_vertex_index]]
                    error_vector = np.subtract(ee_position_target, ee_position_actual)
                    error = np.linalg.norm(error_vector)

            if error > self._error_tolerance:  # No solution found, maxed out iterations
                logger.warning(f"Could not solve IK, maxed out iterations (max: {self._max_iterations})")
                logger.warning(f"Final solution error: {error}")

            else:  # Solution found
                # Calculate joint angles for the solution
                joint_configuration = calculate_joint_angles(vertices=vertices, linear_base=linear_base)
                if linear_base:
                    joint_configuration[0] = link_lengths[0] * np.sign(vertices["x"][1])

                if mirror:
                    # Find the vertices and joint angles for the mirrored solution
                    mirrored_vertices = self._get_mirror_configuration(vertices=vertices,
                                                                       linear_base=linear_base,
                                                                       n_links=n_links)

                    mirrored_joint_configuration = calculate_joint_angles(vertices=mirrored_vertices,
                                                                          linear_base=linear_base)
                    mirrored_joint_configuration[0] = joint_configuration[0]

                self.solved = True
                logger.debug(f"IK solution found in {iterations} iterations.")

                # Update solution dict
                solution["joint_config"] = joint_configuration
                solution["mirrored_joint_config"] = mirrored_joint_configuration
                solution["vertices"] = vertices
                solution["mirrored_vertices"] = mirrored_vertices
                solution["link_lengths"] = link_lengths

        return solution

    def _get_mirror_configuration(self,
                                  vertices: dict[str, list[float]],
                                  linear_base: bool,
                                  n_links: int) -> dict[str, list[float]]:

        mirrored_vertices = copy.deepcopy(vertices)  # Start by copying current vertices

        # The first robot arm vertex is where the mirror line starts
        mirror_line_start_vertex_index = 0 if not linear_base else 1
        mirror_line_start_vertex = [mirrored_vertices["x"][mirror_line_start_vertex_index],
                                    mirrored_vertices["y"][mirror_line_start_vertex_index]]

        # The mirror line ends at the last vertex if theres no orientation, or second last if there is an orientation
        # n_links updates based on if a linear base exists.
        # vertices[n_links] = last vertex, vertices[n_links - 1] = second last vertex
        mirror_line_last_vertex_index = n_links if self._target_orientation is None else n_links - 1
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
