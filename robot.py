import copy
from functools import partial
from typing import Union
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import matplotlib.animation as animation
from matplotlib.patches import Circle
from helper_functions.helper_functions import wrap_angles_to_pi, MoveType, calculate_joint_angles
from trajectories import QuinticPolynomialTrajectory

SCALE_TO_MM = 1000


class Robot:
    def __init__(self, link_lengths: list[float],
                 ik_alg,
                 joint_configuration: list[float] = None,
                 robot_base_radius: float = 0.1,
                 linear_base: bool = False,
                 environment: Union[str, list[float]] = None,
                 trajectory_generator=QuinticPolynomialTrajectory) -> None:

        self.__link_lengths = list(map(lambda x: float(x) * SCALE_TO_MM, link_lengths))  # Scaling links from m to mm
        self.__ik_alg = ik_alg
        self.joint_configuration = wrap_angles_to_pi(joint_configuration)  # Wrap all joint angles
        self.__robot_base_radius = robot_base_radius * SCALE_TO_MM  # Scaling radius from m to mm
        self.__linear_base = linear_base
        self.__environment = environment
        self.__trajectory_generator = trajectory_generator

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

    @link_lengths.setter
    def link_lengths(self, new_lengths):
        self.__link_lengths = new_lengths

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
            self.__n_links += 1  # Update number of links
            self.vertices["x"].insert(0, self.robot_base_origin[0])
            self.vertices["y"].insert(0, self.robot_base_origin[1])

        n_arm_links = self.n_links - 1 if self.linear_base else self.n_links  # Calculate number of arm links

        # If no joint configuration given, create one based on foldability
        # foldable robot starts folded, unfoldable starts outstreched
        # Arm joint configuration is the FK target
        if self.joint_configuration is None:
            self.joint_configuration = [0.0] * self.n_links
            foldable = self.__is_foldable()

            # If foldable, created folded config
            if foldable:
                # Set last n-1 joints through alternating pi, -pi joint angles
                # Then insert first joint angle = 0.0 to have it stretched out
                arm_joint_configuration = list(map(lambda x: ((-1) ** x) * np.pi, range(n_arm_links - 1)))
                arm_joint_configuration.insert(0, 0.0)

            # If not foldable, create outstreched config
            else:
                arm_joint_configuration = [0.0] * n_arm_links

        # If joint configuration given, check if the number given matches number of links
        else:
            assert len(self.joint_configuration) == n_arm_links, \
                f'Number of joint angles ({len(self.joint_configuration)}) ' \
                f'does not equal number of links ({n_arm_links})'
            arm_joint_configuration = self.joint_configuration

        # Call forward kinematics to move the robot to the starting joint_configuration
        self.move(move_type=MoveType.JOINT, plot=False, enable_animation=False, debug=False, target_configuration=arm_joint_configuration)
        self.mirrored_vertices = self.vertices  # Mirrored vertices set to prevent crashes for gui

    def _plot(self, ax: Axes, canvas=None, mirror: bool = False, target_orientation: float = None) -> None:
        ax.cla()
        ax.axis('scaled')
        ax.autoscale(False, "both", tight=True)
        if not self.linear_base:
            ax.set_ylim(-self.robot_base_radius, sum(self.link_lengths))
        else:
            ax.set_ylim(-self.robot_base_radius, sum(self.link_lengths[1::]))

        # Drawing robot
        if not self.linear_base:
            # No linear base so all vertices are robot arm
            current_robot_base = self.robot_base_origin
            ax.plot(self.vertices["x"], self.vertices["y"], 'go-')
            mirror_start_index = 0  # First robot arm vertex to start mirror from is at idx 0
        else:
            # Linear base enabled so current base at end of prismatic link
            current_robot_base = (self.vertices["x"][1], self.vertices["y"][1])

            # Show linear base as dotted blue line
            ax.plot(self.vertices["x"][0:2], self.vertices["y"][0:2], 'bo--')

            # Show robot arm as green line
            ax.plot(self.vertices["x"][1:], self.vertices["y"][1:], 'go-')
            mirror_start_index = 1  # First robot arm vertex to start mirror from is at idx 1

        if not self.linear_base:
            xlim_max = current_robot_base[0] + sum(self.link_lengths)
            xlim_min = current_robot_base[0] - sum(self.link_lengths)
        else:
            xlim_max = current_robot_base[0] + sum(self.link_lengths[1::])
            xlim_min = current_robot_base[0] - sum(self.link_lengths[1::])

        ax.set_xlim(xlim_min, xlim_max)

        # Plot mirrored links
        if mirror:
            # If no target orientation, mirror includes all links from start (accounting for linear base) to end
            if target_orientation is None:
                ax.plot(self.mirrored_vertices["x"][mirror_start_index:],
                        self.mirrored_vertices["y"][mirror_start_index:], 'ro-')

            # If target orientation, mirror only includes links from start (accounting for linear base) to end - 1
            # Because the last link will be coincident for mirror and original
            else:
                ax.plot(self.mirrored_vertices["x"][mirror_start_index:-1],
                        self.mirrored_vertices["y"][mirror_start_index:-1], 'ro-')

        # Draw robot base
        base = Circle(current_robot_base, self.robot_base_radius, color="green")
        ax.add_patch(base)

        if canvas is not None:
            canvas.draw()
            canvas.flush_events()

    def animated_fk_plot(self, target_configuration):
        self.move_fk(target_configuration=target_configuration, debug=False)
        self._plot(plt.gca())

    def animated_ik_plot(self, ik_params, mirror):
        if self.linear_base:
            rail_increment, *target_configuration = ik_params
            self.move_fk(target_configuration=target_configuration, debug=False)
            self.move_rail(rail_increment)
        else:
            target_configuration = ik_params
            self.move_fk(target_configuration=target_configuration, debug=False)
        self._plot(plt.gca(), mirror=mirror)

    def get_trajectory(self, move_type: MoveType, **kwargs):
        traj = []

        # IK cartesian solution is converted into joint space and trajectory calculated from current joint config.
        # Joint space trajectory is smoother.
        if move_type == MoveType.CARTESIAN:
            target_position = kwargs.get('target_position')
            target_orientation = kwargs.get('target_orientation')

            # Solve IK to get the joint configuration at the target position and create joint space trajectory
            ik = self.ik_alg(robot=self, target_position=target_position, target_orientation=target_orientation)
            target_configuration, mirrored_configuration = ik.solve(debug=False, mirror=False, move=False)

            for joint_idx, joint_target in enumerate(target_configuration):
                current_joint_value = self.joint_configuration[joint_idx]
                traj_gen = self.__trajectory_generator((current_joint_value, joint_target))
                _, joint_setpoints = traj_gen.solve_traj()
                traj.append(joint_setpoints)

            traj = list(zip(*traj))

        elif move_type == MoveType.JOINT:
            target_configuration = kwargs.get('target_configuration')
            for joint_idx, joint_target in enumerate(target_configuration):
                if self.linear_base:
                    arm_joint_index = joint_idx + 1
                else:
                    arm_joint_index = joint_idx
                current_joint_val = self.joint_configuration[arm_joint_index]
                traj_gen = self.__trajectory_generator((current_joint_val, joint_target))
                _, joint_setpoints = traj_gen.solve_traj()
                traj.append(joint_setpoints)
            traj = list(zip(*traj))

        return traj

    def move_rail(self, rail_length):
        if not self.linear_base:
            raise Exception("Cant move rail, no rail configured.")
        # Maintain x distances between vertices

        # Shift all vertices by an increment relative to starting position
        old_base_pos = self.vertices["x"][1]
        self.vertices["x"][1] = rail_length + self.robot_base_origin[0]

        relative_change = self.vertices["x"][1] - old_base_pos
        for joint_idx, joint_val in enumerate(self.vertices["x"][2::]):
            self.vertices["x"][joint_idx+1] += relative_change

        new_link_lengths = copy.deepcopy(self.link_lengths)
        new_link_lengths[0] = rail_length
        self.link_lengths = new_link_lengths
        self.joint_configuration[0] = new_link_lengths[0]

    def inverse_kinematics(self, target_position: list[float],
                           target_orientation: float = None,
                           mirror: bool = True,
                           debug: bool = False,
                           move: bool = True,
                           plot: bool = False,
                           enable_animation: bool = True) -> None:

        if not enable_animation:
            # Create ik object and call solve method
            ik = self.ik_alg(robot=self, target_position=target_position, target_orientation=target_orientation)
            ik.solve(debug=debug, mirror=mirror, move=move)

            # If solution to ik found and plot == True, plot the robot
            if ik.solved and plot:
                fig, ax = plt.subplots()
                self._plot(ax=ax, mirror=mirror, target_orientation=target_orientation)

            # Debug provides detailed info, this is more for gui usage since terminal can run with debug=True
            # Not debug condition prevents double printing for terminal
            elif not ik.solved and not debug:
                print("IK cannot be solved. Pick a more appropriate target")

        else:
            ik_traj = self.get_trajectory(move_type=MoveType.CARTESIAN, target_position=target_position, target_orientation=target_orientation)
            fig, ax = plt.subplots()
            animated_ik_plot_func = partial(self.animated_ik_plot, mirror=mirror)
            ani = animation.FuncAnimation(fig, animated_ik_plot_func, ik_traj, interval=1, repeat=False)
            plt.show()

    def move(self,
             move_type: MoveType,
             plot: bool = False,
             enable_animation: bool = True,
             debug: bool = False,
             **kwargs):

        if move_type == MoveType.JOINT:
            target_configuration = kwargs.get("target_configuration")
            target_configuration = wrap_angles_to_pi(target_configuration)
            if not enable_animation:
                self.move_fk(target_configuration=target_configuration, debug=debug)
                if plot:
                    fig, ax = plt.subplots()
                    self._plot(ax=ax)
                    plt.show()
            else:
                target_configuration_traj = self.get_trajectory(move_type=move_type, target_configuration=target_configuration)
                if plot:
                    fig, ax = plt.subplots()
                    ani = animation.FuncAnimation(fig, self.animated_fk_plot, target_configuration_traj, interval=1, repeat=False)
                    plt.show()

        elif move_type == MoveType.CARTESIAN:
            pass

    def move_fk(self, target_configuration, debug: bool = False):
        target_configuration = list(target_configuration)
        # Need to dummy joint angle for the linear prismatic joint to avoid indexing error
        start_idx = 0
        if self.linear_base:
            start_idx = 1  # Skipping first link since the prismatic joint wont be controlled through FK
            # For an FK command we are not concerned with controlling the linear base (prismatic joint)
            target_configuration.insert(0, self.link_lengths[0])

        reference_angle = 0
        for link_index in range(start_idx, self.n_links):
            # Joint angles are computed as relative to previous link (convention)
            behind_vertex = [self.vertices["x"][link_index], self.vertices["y"][link_index]]
            angle = target_configuration[link_index] + reference_angle
            x_change = self.link_lengths[link_index] * np.cos(angle)
            y_change = self.link_lengths[link_index] * np.sin(angle)

            new_vertex = np.add(behind_vertex, [x_change, y_change])

            self.vertices["x"][link_index + 1] = new_vertex[0]
            self.vertices["y"][link_index + 1] = new_vertex[1]

            reference_angle += target_configuration[link_index]
        self.joint_configuration = calculate_joint_angles(self.vertices, self.linear_base)

        if debug:
            print("Final robot configuration:")
            print(self.joint_configuration)
            print(self.vertices)

        return self.vertices
