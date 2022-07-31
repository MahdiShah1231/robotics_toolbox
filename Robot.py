import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from InverseKinematics import Fabrik
from ForwardKinematics import ForwardKinematics
from helper_functions.helper_functions import calculate_joint_angles, wrap_angles_to_pi, draw_environment


class Robot:
    def __init__(self, link_lengths, ik_alg=None, joint_configuration=None, robot_base_radius=None, linear_base=False, environment=None):
        self.ik_alg = ik_alg
        scaled_links = [length*1000 for length in link_lengths]
        self.collisions = False
        self.link_lengths = scaled_links
        self.link_number = len(link_lengths)
        self.joint_configuration = joint_configuration
        self.mirrored_joint_configuration = None
        self.robot_base_radius = robot_base_radius*1000
        self.robot_base_origin = [self.robot_base_radius, 0]
        self.linear_base = linear_base
        self.mirrored_vertices = None
        self.environment = environment
        x_vertices = [0]*(self.link_number+1)
        y_vertices = [0]*(self.link_number+1)

        if ik_alg is None:
            self.ik_alg = Fabrik

        if joint_configuration is None:
            foldable = self.__is_foldable()
            for vertex_number in range(1, len(x_vertices)):
                if foldable: # Starting configuration is folded
                    if vertex_number % 2 != 0:
                        x_vertices[vertex_number] = x_vertices[vertex_number-1] + self.link_lengths[vertex_number-1]
                    else:
                        x_vertices[vertex_number] = x_vertices[vertex_number-1] - self.link_lengths[vertex_number-1]
                else: # Starting configuration is outstretched
                    x_vertices[vertex_number] = x_vertices[vertex_number-1] + self.link_lengths[vertex_number-1]

        else:
            assert len(joint_configuration) == len(link_lengths), f'Number of joint angles ({len(joint_configuration)}), ' \
                                                                  f'does not equal number of links ({len(link_lengths)})'

            self.joint_configuration = wrap_angles_to_pi(self.joint_configuration)
            reference_angle = 0
            for vertex_number in range(1, len(x_vertices)):
                x_vertices[vertex_number] = x_vertices[vertex_number-1] + self.link_lengths[vertex_number-1]*np.cos(self.joint_configuration[vertex_number-1] + reference_angle)
                y_vertices[vertex_number] = y_vertices[vertex_number-1] + self.link_lengths[vertex_number-1]*np.sin(self.joint_configuration[vertex_number-1] + reference_angle)
                reference_angle += self.joint_configuration[vertex_number-1]

        offset_x_vertices = [x+self.robot_base_origin[0] for x in x_vertices]
        self.vertices = {"x": offset_x_vertices, "y": y_vertices}

        if self.linear_base:
            self.link_lengths.insert(0, 0)
            self.link_number = len(self.link_lengths)
            self.vertices["x"].insert(0, self.robot_base_origin[0])
            self.vertices["y"].insert(0, self.robot_base_origin[1])
        if self.joint_configuration is None:
            self.joint_configuration = calculate_joint_angles(self.vertices)

    def __is_foldable(self):
        foldable = True
        for link_index in range(len(self.link_lengths) - 1):
            current_link_length = self.link_lengths[link_index]
            next_link_length = self.link_lengths[link_index + 1]

            if next_link_length > current_link_length:
                foldable = False
                break
        return foldable

    def inverse_kinematics(self, target_position, target_orientation, mirror=False, debug=False):
        ik = self.ik_alg(robot=self, target_position=target_position, target_orientation=target_orientation)
        ik.solve(debug=debug, mirror=mirror)
        if ik.solved:
            self.plot(mirror=mirror, target_orientation=target_orientation)

    def forward_kinematics(self, target_configuration):
        fk = ForwardKinematics(robot=self, target_configuration=target_configuration)
        fk.forward_kinematics()
        self.plot()

    def plot(self, mirror = False, target_orientation = False):
        vertices = self.vertices
        mirrored_vertices = self.mirrored_vertices
        environment = self.environment

        if environment is None:
            draw_environment(robot_base_radius=self.robot_base_radius)
        else:
            if type(environment) == str:
                try:
                    environment_img = Image.open(environment)
                    plt.imshow(environment_img)
                except:
                    raise "Environment error, please ensure environment is a valid .ppm file in the directory"
                else:
                    y_vertices_in_img_frame = [-(y - 850) for y in vertices["y"]]
                    vertices["y"] = y_vertices_in_img_frame
            elif type(environment) == list:
                assert len(environment) == 2, "Environment list must have two elements, [env_width, env_height]"
                draw_environment(self.robot_base_radius, environment[0], environment[1])

        if self.linear_base is False:
            robot_base_origin = (self.robot_base_radius, 0)
            plt.plot(vertices["x"], vertices["y"], 'go-')
            if mirror:
                if target_orientation is None:
                    plt.plot(mirrored_vertices["x"][1:], mirrored_vertices["y"][1:], 'ro-')
                else:
                    plt.plot(mirrored_vertices["x"][:-1], mirrored_vertices["y"][:-1], 'ro-')
        else:
            robot_base_origin = (vertices["x"][1], vertices["y"][1])
            plt.plot(vertices["x"][0:2], vertices["y"][0:2], 'bo--')
            plt.plot(vertices["x"][1:], vertices["y"][1:], 'go-')
            if mirror:
                if target_orientation is None:
                    plt.plot(mirrored_vertices["x"][1:], mirrored_vertices["y"][1:], 'ro-')
                else:
                    plt.plot(mirrored_vertices["x"][1:-1], mirrored_vertices["y"][1:-1], 'ro-')

        base = plt.Circle(robot_base_origin, self.robot_base_radius, color="green")
        plt.gcf().gca().add_artist(base)
        plt.axis('image')
        plt.show()
