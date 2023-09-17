# robotics_toolbox

## 1. What does it do?
This is a simple robotics toolbox that aims to implement and simulate SCARA robot motions using 2D plots. It's mainly a learning project for me to explore different areas of robotics and implement them.  
Currently, the toolbox **only** works with planar SCARA configurations, maybe it will be expanded in the future to have 3D plots of more complex configurations such as 6 DOF manipulators.  
The toolbox allows for simple joint space (forward kinematics) and cartesian space (inverse kinematics) control for the SCARA. There is also the option to add a linear base, which is essentially a transfer unit
to move the robot across the horizontal axis.

## 2. How do I use it?
### 2.1 Setting up the repo
First clone the repository, open it up in your preferred IDE and setup the python environment. I have provided a pipfile and I recommend using a pipenv to install all the requirements.  
To install dependencies with pipenv, first open up the project and type into a python terminal: `pipenv install`  
This will configure the python interpretor with all the requirements used for this toolbox (e.g pyqt, numpy, matplotlib etc...)

### 2.2 Interacting with the interfaces
The toolbox has two different interfaces, a simple python script interface and a PyQt GUI interface. The scripting interface is just a barebones implementation of all the features, 
I suggest using GUI for a better user experience. The GUI allows you to more easily customise the configuration of the SCARA, and control it in a more fluid way. You can quickly
switch between joint space and cartesian space control and see the robot update instantly.

#### 2.2.1 Script interface - main.py
The script interface provides two helper functions which are all that are needed to interact with the robot; `create_robot()`, and `move()`. To use it, just run the script after following the steps. 
First start out by configuring the robot parameters from line 46.  
`robot_link_lengths` - This will be the link lengths of the SCARA arm, the length of the list will give the number of links.  
`robot_base_radius` - This will be the radius of the circular base.  
`starting_joint_configuration`  - [OPTIONAL] This is the starting joint configuration for the SCARA. If its left as `NONE`, defaults will be calculated to be folded if possible or otherwise outstretched.  
`linear_base` - This will enable the usage of a robot transfer unit, to move the robot across the horizontal axis.  
`ik_solver` - This sets the IK Solver for the Inverse Kinematics calculations.  
`trajectory_generator` - This sets the trajectory generator for the animated motions.  

Once that's done, use the helper function `create_robot()` to create a robot with the previously configured parameters. This should be done already line 54 with the arguments being passed as the 
variables configured from the last step. 

##### 2.2.1.1 Joint Space Motions (Forward Kinematics)
Relevant motion type:  `MoveType.JOINT`  

To send joint space commands, use the `move()` function with the following keyword arguments:  
`robot=r` - This is the robot object created  
`move_type = MoveType.JOINT` - This specifies a joint space motion  
`target_configuration = list[float]`  - Joint targets in radains. Please note the length must be equal to the number of links of the robot (set when configuring link lengths). The first element will be the target
for joint 0, second element for joint 1 etc..

Example:  
`move(robot=r, move_type=MoveType.JOINT, target_configuration=[1.57, 0.0, 0.0])`  - This target should point the SCARA directly up.

##### 2.2.1.2 Cartesian Space Motions (Forward Kinematics)
Relevant motion type:  `MoveType.CARTESIAN`

To send joint space commands, use the `move()` function with the following keyword arguments:  
`robot=r` - This is the robot object created  
`move_type = MoveType.CARTESIAN` - This specifies a joint space motion  
`target_position = list[float]`  - Cartesian space target coordinates (in meters).  
`target_orientation = float`  - [OPTIONAL] This will be the target orientation (in radians) to achieve. The IK will have many solutions and in most cases can reach a given target from multiple approach angles.
This will allow you to define the angle to approach from. `target_orientation = np.pi / 2` will reach from below.  
`mirror = bool` - [OPTIONAL] This will allow the calculation of alternative IK configurations that arise from symmetrical solutions with an "elbow" flipping either side of the symmetry line. NOTE: This feature
is currently broken due to changes in how the IK is used. It will be reimplemented in the future.

Example:  
`move(robot=r, move_type=MoveType.CARTESIAN, target_position = [0.5, 0.6], target_orientation = np.pi / 2)`  - This target should point the SCARA directly up.

#### 2.2.2 PyQt GUI interface - main_gui.py
The GUI is a bit more straightforward to use, run the main_gui script and configure the robot by filling out the parameters. Use the tooltips to guide you by hovering over the boxes.  

Once finished with configuring, press the `Create Robot` button to launch the control window.

##### 2.2.2.1 Joint Space Motions (Forward Kinematics)
To send joint space commands, edit the boxes in the upper right corner of the window. Note: the number of joint boxes here should equal the number of comma separated values given for the previous window's link
lengths. These values will correspond to the joint targets for the respective joint, in radians.  

Once the values are set, press the `Go (FK)` button.  

##### 2.2.2.2 Cartesian Space Motions (Forward Kinematics)
To send cartesian space commands, edit the boxes in the lower right side of the window.  
`Target Position (x,y)` - The left is for the x coordinate and the right is for the y coordinate, both in meters.  
`Target Orientation` - Specifies the approach angle for the IK solution. A value of `np.pi / 2` will reach from below.
`Mirror` - This will allow the calculation of alternative IK configurations that arise from symmetrical solutions with an "elbow" flipping either side of the symmetry line. NOTE: This feature
is currently broken due to changes in how the IK is used. It will be reimplemented in the future.  

Once the values are set, press the `Go (IK)` button.

Note:
Alternatively, you can use the **left mouse button** to send IK targets by clicking on the figure canvas. This can be used in conjunction with setting a target orientation in the box to 
send click commands with a fixed target orientation.
