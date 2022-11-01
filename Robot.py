import numpy as np

from InverseKinematics import Fabrik


class Robot:
    def __init__(self, link_lengths, ik_alg=None, joint_configuration=None, robot_base_radius=None, linear_base=False):
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
            for vertex_number in range(1, len(x_vertices)):
                x_vertices[vertex_number] = x_vertices[vertex_number-1] + self.link_lengths[vertex_number-1]*np.cos(self.joint_configuration[vertex_number-1])
                y_vertices[vertex_number] = y_vertices[vertex_number-1] + self.link_lengths[vertex_number-1]*np.sin(self.joint_configuration[vertex_number-1])

        offset_x_vertices = [x+self.robot_base_origin[0] for x in x_vertices]
        self.vertices = {"x": offset_x_vertices, "y": y_vertices}

        if self.linear_base:
            self.link_lengths.insert(0, 0)
            self.link_number = len(self.link_lengths)
            self.vertices["x"].insert(0, self.robot_base_origin[0])
            self.vertices["y"].insert(0, self.robot_base_origin[1])

    def __is_foldable(self):
        foldable = True
        for link_index in range(len(self.link_lengths) - 1):
            current_link_length = self.link_lengths[link_index]
            next_link_length = self.link_lengths[link_index + 1]

            if next_link_length > current_link_length:
                foldable = False
                break
        return foldable

    def inverse_kinematics(self, target_position, target_orientation, environment=None, mirror=False, debug=False):
        fabrik = Fabrik(robot=self, target_position=target_position, target_orientation=target_orientation)
        fabrik.solve(debug=debug)
        if fabrik.solved:
            fabrik.plot(environment=environment, mirror=mirror)
