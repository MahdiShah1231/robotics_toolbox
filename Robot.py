import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from InverseKinematics import Fabrik
from ForwardKinematics import ForwardKinematics
from helper_functions.helper_functions import wrap_angles_to_pi, draw_environment


class Robot:
    def __init__(self, link_lengths,
                 ik_alg=None,
                 joint_configuration=None,
                 robot_base_radius=None,
                 linear_base=False,
                 environment=None):
        self.link_lengths = list(map(lambda x: x * 1000, link_lengths))  # Scaling links from m to mm
        self.ik_alg = ik_alg
        self.joint_configuration = wrap_angles_to_pi(joint_configuration)
        self.robot_base_radius = robot_base_radius * 1000  # Scaling radius from m to mm
        self.linear_base = linear_base
        self.environment = environment

        self.n_links = len(link_lengths)
        self.vertices = {"x": [0] * (self.n_links + 1), "y": [0] * (self.n_links + 1)}
        self.robot_base_origin = [self.robot_base_radius, 0]
        self.mirrored_joint_configuration = None
        self.mirrored_vertices = None
        self.collisions = False

        if ik_alg is None:
            self.ik_alg = Fabrik

        # Shifting vertices to origin of robot base
        self.vertices["x"] = list(map(lambda x: x + self.robot_base_radius, self.vertices["x"]))

        if self.linear_base:
            self.link_lengths.insert(0, 0)
            self.n_links = len(self.link_lengths)
            self.vertices["x"].insert(0, self.robot_base_origin[0])
            self.vertices["y"].insert(0, self.robot_base_origin[1])

        if joint_configuration is None:
            foldable = self.__is_foldable()
            if self.linear_base:
                n_arm_links = self.n_links - 1  # Excluding linear base prismatic link
            else:
                n_arm_links = self.n_links

            if foldable:
                joint_configuration = list(map(lambda x: ((-1) ** x) * np.pi, range(n_arm_links - 1)))
                joint_configuration.insert(0, 0)  # First joint outstretched
            else:
                joint_configuration = [0] * n_arm_links
            self.joint_configuration = joint_configuration
        else:
            assert len(joint_configuration) == len(link_lengths), f'Number of joint angles ({len(joint_configuration)}), ' \
                                                                  f'does not equal number of links ({len(link_lengths)})'
        self.forward_kinematics(target_configuration=joint_configuration, initialise=True, plot=False)

    def __is_foldable(self):
        foldable = True
        if not self.linear_base:
            start = 0
        else:
            start = 1
        link_index = start
        while foldable and link_index in range(start, self.n_links - 1):
            current_link_length = self.link_lengths[link_index]
            next_link_length = self.link_lengths[link_index + 1]

            if next_link_length > current_link_length:
                foldable = False
            link_index += 1
        return foldable

    def __plot(self, mirror=False, target_orientation=False):
        vertices = self.vertices
        mirrored_vertices = self.mirrored_vertices
        environment = self.environment

        if environment is None:
            draw_environment(robot_base_radius=self.robot_base_radius)
        else:
            if type(environment) == str:
                # try:
                #     environment_img = Image.open(environment)
                #     plt.imshow(environment_img)
                # except:
                #     raise "Environment error, please ensure environment is a valid .ppm file in the directory"
                # else:
                #     y_vertices_in_img_frame = [-(y - 850) for y in vertices["y"]]
                #     vertices["y"] = y_vertices_in_img_frame
                raise Exception(".ppm support not implemented yet")
            elif type(environment) == list:
                assert len(environment) == 2, "Environment list must have two elements, [env_width, env_height]"
                draw_environment(self.robot_base_radius, environment[0], environment[1])

        if self.linear_base is False:
            robot_base_origin = self.robot_base_origin
            plt.plot(vertices["x"], vertices["y"], 'go-')
            mirror_start_index = 0
        else:
            robot_base_origin = (vertices["x"][1], vertices["y"][1])
            plt.plot(vertices["x"][0:2], vertices["y"][0:2], 'bo--')
            plt.plot(vertices["x"][1:], vertices["y"][1:], 'go-')
            mirror_start_index = 1

        if mirror:
            if target_orientation is None:
                plt.plot(mirrored_vertices["x"][mirror_start_index:],
                         mirrored_vertices["y"][mirror_start_index:], 'ro-')
            else:
                plt.plot(mirrored_vertices["x"][mirror_start_index:-1],
                         mirrored_vertices["y"][mirror_start_index:-1], 'ro-')

        base = plt.Circle(robot_base_origin, self.robot_base_radius, color="green")
        plt.gcf().gca().add_artist(base)
        plt.axis('image')
        plt.show()

    def inverse_kinematics(self, target_position, target_orientation, mirror=True, debug=False):
        ik = self.ik_alg(robot=self, target_position=target_position, target_orientation=target_orientation)
        ik.solve(debug=debug, mirror=mirror)
        if ik.solved:
            self.__plot(mirror=mirror, target_orientation=target_orientation)

    def forward_kinematics(self, target_configuration, initialise=False, plot=True):
        fk = ForwardKinematics(robot=self, target_configuration=target_configuration)
        fk.forward_kinematics(initialise=initialise)
        if plot:
            self.__plot()
