from typing import List, Any, Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from InverseKinematics import Fabrik
from ForwardKinematics import ForwardKinematics
from helper_functions.helper_functions import wrap_angles_to_pi, draw_environment

SCALE_TO_MM = 1000


class Robot:
    def __init__(self, link_lengths: List[float],
                 ik_alg: Any = Fabrik,
                 joint_configuration: Optional[List[float]] = None,
                 robot_base_radius: float = 0.1,
                 linear_base: bool = False,
                 environment: Union[str, List[float]] = None) -> None:

        self.__link_lengths = list(map(lambda x: float(x) * SCALE_TO_MM, link_lengths))  # Scaling links from m to mm
        self.__ik_alg = ik_alg
        self.joint_configuration = wrap_angles_to_pi(joint_configuration)  # Wrap all joint angles
        self.__robot_base_radius = robot_base_radius * SCALE_TO_MM  # Scaling radius from m to mm
        self.__linear_base = linear_base
        self.__environment = environment

        # Generated attributes
        self.__robot_base_origin = [self.robot_base_radius, 0.0]
        self.__n_links = len(link_lengths)
        self.vertices = {"x": [0.0] * (self.n_links + 1), "y": [0.0] * (self.n_links + 1)}
        self.vertices["x"] = list(map(lambda x: x + self.robot_base_radius, self.vertices["x"]))
        self.mirrored_joint_configuration = None
        self.mirrored_vertices = None
        self.collisions = False

        # Initial robot configuration
        self.__configure_robot()

    @property
    def link_lengths(self):
        return self.__link_lengths

    @property
    def ik_alg(self):
        return self.__ik_alg

    @property
    def robot_base_radius(self):
        return self.__robot_base_radius

    @property
    def linear_base(self):
        return self.__linear_base

    @property
    def environment(self):
        return self.__environment

    @property
    def n_links(self):
        return self.__n_links

    @property
    def robot_base_origin(self):
        return self.__robot_base_origin

    def __is_foldable(self) -> bool:
        foldable = True
        # linear base adds an extra "prismatic" link at base, exclude this for foldability check
        start_link_idx = 0 if not self.linear_base else 1
        link_index = start_link_idx

        # Iterate through links, check if each successive link gets smaller. Stop on first false result
        while foldable and link_index in range(start_link_idx, self.n_links - 1):
            current_link_length = self.link_lengths[link_index]
            next_link_length = self.link_lengths[link_index + 1]
            if next_link_length > current_link_length:
                foldable = False

            link_index += 1

        return foldable

    def __configure_robot(self) -> None:
        if self.linear_base:
            self.__link_lengths.insert(0, 0.0)  # Insert linear base prismatic link
            self.__n_links = len(self.link_lengths)  # Update number of links
            self.vertices["x"].insert(0, self.robot_base_origin[0])
            self.vertices["y"].insert(0, self.robot_base_origin[1])

        n_arm_links = self.n_links - 1 if self.linear_base else self.n_links  # Calculate number of arm links

        # If no joint configuration given, create one based on foldability
        # foldable robot starts folded, unfoldable starts outstreched
        if self.joint_configuration is None:
            foldable = self.__is_foldable()

            # If foldable, created folded config
            if foldable:
                # Set last n-1 joints through alternating pi, -pi joint angles
                # Then insert first joint angle = 0.0 to have it stretched out
                joint_configuration = list(map(lambda x: ((-1) ** x) * np.pi, range(n_arm_links - 1)))
                joint_configuration.insert(0, 0.0)

            # If not foldable, create outstreched config
            else:
                joint_configuration = [0.0] * n_arm_links

            self.joint_configuration = joint_configuration

        # If joint configuration given, check if the number given matches number of links
        else:
            assert len(self.joint_configuration) == n_arm_links, \
                f'Number of joint angles ({len(self.joint_configuration)}) ' \
                f'does not equal number of links ({self.n_links})'

        # Call forward kinematics to move the robot to the starting joint_configuration
        self.forward_kinematics(target_configuration=self.joint_configuration, debug=False, plot=False)
        self.mirrored_vertices = self.vertices  # Mirrored vertices set to prevent crashes for gui

    def __plot(self, mirror: bool = False, target_orientation: float = None) -> None:
        vertices = self.vertices
        mirrored_vertices = self.mirrored_vertices
        environment = self.environment

        # If a ppm image is not given for the background of the robot
        if environment is None:
            draw_environment(robot_base_radius=self.robot_base_radius)  # Draw a default environment
        else:
            # Reading image file not implemented yet
            if isinstance(environment, str):
                # try:
                #     environment_img = Image.open(environment)
                #     plt.imshow(environment_img)
                # except:
                #     raise "Environment error, please ensure environment is a valid .ppm file in the directory"
                # else:
                #     y_vertices_in_img_frame = [-(y - 850) for y in vertices["y"]]
                #     vertices["y"] = y_vertices_in_img_frame
                raise NotImplementedError(".ppm support not implemented yet")
            # Creating environment given width and height
            elif isinstance(environment, list):
                assert len(environment) == 2, "Environment list must have two elements, [env_width, env_height]"
                draw_environment(self.robot_base_radius, environment[0], environment[1])
            else:
                raise ValueError("Environment must be a file path for an image (Not supported yet) "
                                 "or a list containing width and height")

        # Drawing robot
        if not self.linear_base:
            # No linear base so all vertices are robot arm
            current_robot_base = self.robot_base_origin
            plt.plot(vertices["x"], vertices["y"], 'go-')
            mirror_start_index = 0  # First robot arm vertex to start mirror from is at idx 0
        else:
            # Linear base enabled so current base at end of prismatic link
            current_robot_base = (vertices["x"][1], vertices["y"][1])

            # Show linear base as dotted blue line
            plt.plot(vertices["x"][0:2], vertices["y"][0:2], 'bo--')

            # Show robot arm as green line
            plt.plot(vertices["x"][1:], vertices["y"][1:], 'go-')
            mirror_start_index = 1  # First robot arm vertex to start mirror from is at idx 1

        # Plot mirrored links
        if mirror:
            # If no target orientation, mirror includes all links from start (accounting for linear base) to end
            if target_orientation is None:
                plt.plot(mirrored_vertices["x"][mirror_start_index:],
                         mirrored_vertices["y"][mirror_start_index:], 'ro-')

            # If target orientation, mirror only includes links from start (accounting for linear base) to end - 1
            # Because the last link will be coincident for mirror and original
            else:
                plt.plot(mirrored_vertices["x"][mirror_start_index:-1],
                         mirrored_vertices["y"][mirror_start_index:-1], 'ro-')

        # Draw robot base
        base = plt.Circle(current_robot_base, self.robot_base_radius, color="green")
        plt.gcf().gca().add_artist(base)
        plt.axis('image')
        plt.show()

    def inverse_kinematics(self, target_position: List[float],
                           target_orientation: Optional[float] = None,
                           mirror: bool = True,
                           debug: bool = False,
                           plot: bool = False) -> None:

        # Create ik object and call solve method
        ik = self.ik_alg(robot=self, target_position=target_position, target_orientation=target_orientation)
        ik.solve(debug=debug, mirror=mirror)

        # If solution to ik found and plot == True, plot the robot
        if ik.solved and plot:
            self.__plot(mirror=mirror, target_orientation=target_orientation)

        # Debug provides detailed info, this is more for gui usage since terminal can run with debug=True
        # Not debug condition prevents double printing for terminal
        elif not ik.solved and not debug:
            print("IK cannot be solved. Pick a more appropriate target")

    def forward_kinematics(self, target_configuration: List[float],
                           debug: bool = False,
                           plot: bool = False) -> None:

        # Create fk object and call method
        fk = ForwardKinematics(robot=self, target_configuration=target_configuration)
        fk.forward_kinematics(debug=debug)

        if plot:
            self.__plot()
