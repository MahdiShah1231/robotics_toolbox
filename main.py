import numpy as np
from InverseKinematics import Fabrik
from helper_functions.helper_functions import create_robot, forward_kinematics, inverse_kinematics

if __name__ == '__main__':
    r = create_robot(link_lengths=[0.4, 0.3, 0.2],
                     ik_alg=Fabrik,
                     joint_configuration=None,
                     robot_base_radius=0.1,
                     linear_base=True,
                     environment=None)
    # inverse_kinematics(robot=r, target_position=[1, 0.6], target_orientation=np.pi / 2, mirror=True, debug=True)
    # forward_kinematics(robot=r, target_configuration=[0, np.pi/2, -np.pi/2])
    # forward_kinematics(robot=r, target_configuration=[0, 0, 0])
