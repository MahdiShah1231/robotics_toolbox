import logging
import sys
from functools import partial

import matplotlib
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QComboBox, QFormLayout, QRadioButton, QVBoxLayout, \
    QPushButton, QMainWindow, QHBoxLayout, QBoxLayout, QDoubleSpinBox, QLabel
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from helper_functions.helper_functions import MoveType, create_logger
from helper_functions.ik_options import ik_solvers
from helper_functions.trajectory_generator_options import trajectory_generators
from inverse_kinematics import FabrikSolver
from robot import Robot
from trajectories import QuinticPolynomialTrajectory

matplotlib.use('Qt5Agg')
logger = create_logger(module_name=__name__, level=logging.INFO)  # Change debug level as needed


class ConfigureRobotWindow(QMainWindow):
    """A QMainWindow to configure the robot class."""

    def __init__(self) -> None:
        """Initialises the configure robot window."""
        super().__init__(parent=None)
        self.setWindowTitle("ConfigureRobot")

        # Defaults
        self.__link_lengths = [0.4, 0.3, 0.2]
        self.__ik_solver = FabrikSolver()
        self.__joint_configuration = None
        self.__robot_base_radius = 0.1
        self.__linear_base = True
        self.__trajectory_generator = QuinticPolynomialTrajectory()

        self._create_central_widget()
        self._create_menu()

    def _create_central_widget(self) -> None:
        """Create the central widget with all the entry fields."""
        window = QWidget()
        self.setFixedSize(400, 260)
        main_window_layout = QVBoxLayout()
        create_robot_button = QPushButton("Create Robot")
        window.setLayout(main_window_layout)
        form_layout = QFormLayout()

        data_fields = {
            "link_lengths": QLineEdit(),
            "joint_configuration": QLineEdit(),
            "robot_base_radius": QDoubleSpinBox(),
            "ik_solver": QComboBox(),
            "trajectory_generator": QComboBox(),
            "linear_base": QRadioButton("On"),
        }

        # Tooltips
        data_fields["link_lengths"].setToolTip("The link lengths for each joint as comma separated values.")
        data_fields["joint_configuration"].setToolTip("The starting joint angles "
                                                      "for each joint as comma separated values. Can be left empty.")
        data_fields["robot_base_radius"].setToolTip("The radius of the robots circular base.")
        data_fields["ik_solver"].setToolTip("The inverse kinematics solver for cartesian space control.")
        data_fields["trajectory_generator"].setToolTip("The trajectory generator for "
                                                       "animated motions between waypoints.")
        data_fields["linear_base"].setToolTip("The active state for a linear rail to allow horizontal displacement"
                                              " of the robot base.")

        # Configuring QWidgets
        data_fields["link_lengths"].setText(','.join(str(val) for val in self.__link_lengths))  # Setting defaults
        data_fields["ik_solver"].addItem("Fabrik")
        data_fields["trajectory_generator"].addItem("Quintic Polynomial")
        data_fields["robot_base_radius"].setMinimum(self.__robot_base_radius)
        data_fields["robot_base_radius"].setMaximum(1.0)
        data_fields["robot_base_radius"].setSingleStep(0.1)
        data_fields["linear_base"].setChecked(self.__linear_base)
        environment = None  # TODO Implement

        form_layout.addRow("Link Lengths (m):", data_fields["link_lengths"])
        form_layout.addRow("Starting Joint Configuration (rad):", data_fields["joint_configuration"])
        form_layout.addRow("Robot Base Radius (m):", data_fields["robot_base_radius"])
        form_layout.addRow("Inverse Kinematics Solver:", data_fields["ik_solver"])
        form_layout.addRow("Trajectory Generator:", data_fields["trajectory_generator"])
        form_layout.addRow("Linear Base:", data_fields["linear_base"])

        main_window_layout.addLayout(form_layout)
        main_window_layout.addWidget(create_robot_button)

        self.setCentralWidget(window)
        self._connect_signals(data_fields, create_robot_button)

    def _create_menu(self) -> None:
        """Create a menu bar."""
        # TODO implement features
        menu = self.menuBar().addMenu("&Menu")
        menu.addAction("&Exit", self.close)

    def _connect_signals(self, data_fields: dict, button: QPushButton) -> None:
        """Connect Qt signals and slots.

        The signals from the QWidgets used to gather information about the robot configuration are each connected to a
        slot.

        Args:
            data_fields: A dictionary containing the QWidgets used to gather robot configuration information.
            button: The QPushButton used to launch the control window.
        """
        button.clicked.connect(self.launch_robot_control)

        for field_name, field_obj in data_fields.items():
            process_func = partial(self._process_field, field_name, field_obj)

            if isinstance(field_obj, QRadioButton):
                field_obj.toggled.connect(process_func)

            elif isinstance(field_obj, QComboBox):
                field_obj.activated.connect(process_func)

            elif isinstance(field_obj, QLineEdit):
                field_obj.editingFinished.connect(process_func)

            elif isinstance(field_obj, QDoubleSpinBox):
                field_obj.valueChanged.connect(process_func)

            else:
                logger.warning(f"Unimplemented field: {field_name}")

    def _process_field(self, field_name: str, field_obj) -> None:
        """Process signal from the robot configuration QWidget.

        Args:
            field_name: Signal emitting QWidget's name in the data fields dictionary.
            field_obj: Signal emitting QWidget object.
        """
        if isinstance(field_obj, QLineEdit):
            field_value = field_obj.text()

            if field_name == "link_lengths":
                try:
                    self.__link_lengths = [float(length) for length in field_value.split(",")]
                except Exception:
                    logger.error("Error processing link lengths", exc_info=True)
                    logger.error("Must be comma separated numbers.")

            elif field_name == "joint_configuration":
                try:
                    self.__joint_configuration = [float(angle) for angle in field_value.split(",")]
                except Exception:
                    logger.error("Error processing joint configuration", exc_info=True)
                    logger.error("Must be comma separated numbers.")

            else:
                logger.warning(f"Unimplemented QLineEdit: {field_name}. Value: {field_value}")

        elif isinstance(field_obj, QDoubleSpinBox):
            field_value = field_obj.value()

            if field_name == "robot_base_radius":
                self.__robot_base_radius = field_value

            else:
                logger.warning(f"Unimplemented QDoubleSpinBox: {field_name}. Value: {field_value}")

        elif isinstance(field_obj, QComboBox):
            field_value = field_obj.currentText()

            if field_name == "ik_solver":
                self.__ik_solver = ik_solvers[field_value]()

            elif field_name == "trajectory_generator":
                self.__trajectory_generator = trajectory_generators[field_value]()

            else:
                logger.warning(f"Unimplemented QComboBox: {field_name}. Value: {field_value}")

        elif isinstance(field_obj, QRadioButton):
            field_value = field_obj.isChecked()

            if field_name == "linear_base":
                self.__linear_base = field_value

            else:
                logger.warning(f"Unimplemented QRadioButton: {field_name}. Value: {field_value}")

        else:
            logger.warning(f"Unimplemented QObject: {field_name}. Type: {type(field_obj)}")

    def launch_robot_control(self) -> None:
        """Create a robot object and launch the robot control window."""
        robot = Robot(link_lengths=self.__link_lengths,
                      ik_solver=self.__ik_solver,
                      trajectory_generator=self.__trajectory_generator,
                      joint_configuration=self.__joint_configuration,
                      robot_base_radius=self.__robot_base_radius,
                      linear_base=self.__linear_base)
        control_window = ControlWindow(robot)
        control_window.show()


class ControlWindow(QWidget):
    """A control window to interact with the robotic manipulator."""

    def __init__(self, robot: Robot) -> None:
        """Initialise the control window to interact with the robot object.

        Args:
            robot: An initialised Robot object.
        """
        super().__init__()
        self.__robot = robot
        n_arm_joints = self.__robot.n_links - 1 if self.__robot.linear_base else self.__robot.n_links
        self.__fk_joint_targets = [0.0] * n_arm_joints
        self.__ik_target_position = [0.0, 0.0]
        self.__ik_target_orientation = None
        self.__mirror = False
        self.__canvas = None

        # Setting up timer to update GUI
        self.__timer = QTimer()
        self.__timer.setInterval(100)
        self.__timer.timeout.connect(self._update_canvas_dynamic)
        self.__animation_func = None  # Method to animate trajectory (IK or FK)
        self.__traj = []  # List of setpoints to traverse for active trajectory

        self._create_window(n_arm_joints=n_arm_joints)

    def _create_window(self, n_arm_joints: int) -> None:
        """Create the window with all the necessary QWidgets.

        Args:
            n_arm_joints: The number of joints in the articulated arm of the robot.
        """
        main_layout = QHBoxLayout()
        layout = QVBoxLayout()
        self.setLayout(main_layout)
        self.setWindowTitle("ControlPanel")

        self._create_visual_canvas(main_layout)

        buttons = {
            "go_fk": QPushButton("Go (FK)"),
            "go_ik": QPushButton("Go (IK)"),
            "enable_ik_orientation": QPushButton("Enable IK target orientation")
        }
        buttons["enable_ik_orientation"].setCheckable(True)
        buttons["enable_ik_orientation"].setChecked(False)

        fk_form_layout = QFormLayout()
        fk_data_fields = {}
        for joint_idx in range(n_arm_joints):
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-3.14, 3.14)
            spinbox.setSingleStep(0.01)
            spinbox.setSuffix(" rad")
            fk_data_fields[f"Joint {joint_idx}"] = spinbox
            fk_form_layout.addRow(f"Joint {joint_idx}:", fk_data_fields[f"Joint {joint_idx}"])
        fk_form_layout.addRow(buttons["go_fk"])

        ik_form_layout = QFormLayout()
        ik_data_fields = {
            "Target Position x": QDoubleSpinBox(),
            "Target Position y": QDoubleSpinBox(),
            "Target Orientation": QDoubleSpinBox(),
            "Mirror": QRadioButton(),
        }

        ik_data_fields["Target Position x"].setSingleStep(0.1)
        ik_data_fields["Target Position x"].setSuffix(" m")
        ik_data_fields["Target Position y"].setSingleStep(0.1)
        ik_data_fields["Target Position y"].setSuffix(" m")
        ik_data_fields["Target Orientation"].setRange(-3.14, 3.14)
        ik_data_fields["Target Orientation"].setSingleStep(0.1)
        ik_data_fields["Target Orientation"].setSuffix(" rad")
        ik_data_fields["Target Orientation"].setEnabled(False)

        target_position_layout = QHBoxLayout()
        target_position_layout.addWidget(ik_data_fields["Target Position x"])
        target_position_layout.addWidget(ik_data_fields["Target Position y"])

        ik_form_layout.addRow("Target Position (x,y):", target_position_layout)
        ik_form_layout.addRow("Target Orientation:", ik_data_fields["Target Orientation"])
        ik_form_layout.addRow("Mirror:", ik_data_fields["Mirror"])
        ik_form_layout.addRow(buttons["enable_ik_orientation"])
        ik_form_layout.addRow(buttons["go_ik"])

        coordinate_space_heading_font = QFont()
        coordinate_space_heading_font.setBold(True)
        joint_space_label = QLabel("Joint space commands:")
        joint_space_label.setFont(coordinate_space_heading_font)
        layout.addWidget(joint_space_label)
        layout.addLayout(fk_form_layout)
        layout.addStretch(50)
        cartesian_space_label = QLabel("Cartesian space commands:")
        cartesian_space_label.setFont(coordinate_space_heading_font)
        layout.addWidget(cartesian_space_label)
        layout.addLayout(ik_form_layout)
        layout.addStretch(250)
        main_layout.addLayout(layout)
        self.setMinimumSize(1000, 600)
        self._connect_signals(fk_data_fields, ik_data_fields, buttons)

    def _create_visual_canvas(self, layout: QBoxLayout) -> None:
        """Create the Qt visual canvas to plot the robot onto.

        Args:
            layout: The main window QBoxLayout.
        """
        self.__canvas = VisualCanvas(self)
        layout.addWidget(self.__canvas, stretch=1)

        # Allowing click commands to send IK to mouse click
        self.__canvas.mpl_connect('button_press_event', self._on_click)

        # Start QTimer to update visual canvas
        self.__timer.start()

    def _on_click(self, event) -> None:
        """Handle mouse click events on the Qt Canvas.

        Args:
            event: The Qt mouse event.
        """
        if event.button == MouseButton.LEFT:
            self.__ik_target_position = [event.xdata/1000.0, event.ydata/1000.0]
            logger.debug(f"Click event received. Target: {self.__ik_target_position}")
            self._get_traj(move_type=MoveType.CARTESIAN, target_position=self.__ik_target_position,
                           target_orientation=self.__ik_target_orientation)

        elif event.button == MouseButton.RIGHT:
            logger.warning("Context menu to be implemented")  # TODO implement context menu

    # TODO implement click and drag
    def _update_canvas_dynamic(self) -> None:
        """Update the Qt Canvas with the new robot state."""
        if len(self.__traj) != 0:
            while len(self.__traj) > 0:
                config = self.__traj.pop(0)
                self.__animation_func(config)
            logger.info("Trajectory complete.")
            logger.info(f"Final joint configuration: {self.__robot.joint_configuration}")
            logger.info(f"Final vertices: {self.__robot.vertices}")
        else:
            self.__robot._plot(ax=self.__canvas.axes, canvas=self.__canvas, mirror=self.__mirror)

    def _get_traj(self, move_type: MoveType, **kwargs) -> None:
        """Calculate the trajectory in the specified coordinate space.

        Wrapper function around the robot.get_trajectory() method. Also sets the animation_func attribute.

        Args:
            move_type: MoveType enum specifying the coordinate space of the motion command.
            **kwargs: Additional keyword arguments relevant to the specified motion type. Accepts target_position and
                target_orientation for MoveType.CARTESIAN, and accepts target_configuration for MoveType.JOINT.
        """
        if move_type == MoveType.JOINT:
            self.__animation_func = partial(self.__robot.move_fk_animated, ax=self.__canvas.axes, canvas=self.__canvas)
        elif move_type == MoveType.CARTESIAN:
            self.__animation_func = partial(self.__robot.move_ik_animated, ax=self.__canvas.axes, canvas=self.__canvas, mirror=False)
        self.__traj = self.__robot.get_trajectory(move_type, **kwargs)

    def _connect_signals(self, fk_data_fields: dict, ik_data_fields: dict, buttons: dict) -> None:
        """Connect Qt signals and slots.

        Args:
            fk_data_fields: A dictionary containing the QWidgets used to specify a joint space command.
            ik_data_fields: A dictionary containing the QWidgets used to specify a cartesian space command.
            buttons: A dictionary containing the QPushButtons used to send joint space and cartesian space commands.
        """
        # Processing the FK data fields
        for joint_name, joint_field_obj in fk_data_fields.items():
            process_func = partial(self._process_field, joint_name, joint_field_obj)
            joint_field_obj.valueChanged.connect(process_func)

        # Processing the IK data fields
        for field_name, field_obj in ik_data_fields.items():
            process_func = partial(self._process_field, field_name, field_obj)

            if isinstance(field_obj, QRadioButton):
                field_obj.toggled.connect(process_func)

            elif isinstance(field_obj, QLineEdit):
                field_obj.editingFinished.connect(process_func)

            elif isinstance(field_obj, QDoubleSpinBox):
                field_obj.valueChanged.connect(process_func)

            else:
                logger.warning(f"Unimplemented field: {field_name}")

        buttons["go_fk"].clicked.connect(lambda: self._get_traj(move_type=MoveType.JOINT,
                                                                target_configuration=self.__fk_joint_targets))

        buttons["go_ik"].clicked.connect(lambda: self._get_traj(move_type=MoveType.CARTESIAN,
                                                                target_position=self.__ik_target_position,
                                                                target_orientation=self.__ik_target_orientation))

        buttons["enable_ik_orientation"].clicked.connect(
            lambda: self._toggle_ik_target_orientation(ik_data_fields["Target Orientation"],
                                                       button=buttons["enable_ik_orientation"])
        )

    def _toggle_ik_target_orientation(self, target_orientation_field: QDoubleSpinBox, button: QPushButton):
        """Toggle the ik target orientation.

        This also enables/disables the target orientation field. The button must be toggled to edit the field.
        If the button is toggled, the target orientation is sent, otherwise the IK will approach at any angle.

        Args:
            target_orientation_field: The QDoubleSpinBox containing the IK target orientation information.
            button: The QPushButton emitting the signal.
        """
        if button.isChecked():
            # Target orientation is enabled.
            target_orientation_field.setEnabled(True)
            self.__ik_target_orientation = target_orientation_field.value()
        else:
            # Target orientation is disabled.
            target_orientation_field.setEnabled(False)
            self.__ik_target_orientation = None

    def _process_field(self, field_name: str, field_obj) -> None:
        """Process signal from the emitting QWidget.

        Args:
            field_name: Signal emitting QWidget's name in the data fields dictionary.
            field_obj: Signal emitting QWidget object.
        """
        if isinstance(field_obj, QDoubleSpinBox):
            field_value = field_obj.value()

            if field_name == 'Target Position x':
                self.__ik_target_position[0] = field_value

            elif field_name == 'Target Position y':
                self.__ik_target_position[1] = field_value

            elif field_name == 'Target Orientation':
                self.__ik_target_orientation = field_value

            elif "Joint" in field_name:  # Field name = "Joint" + idx, for fk target
                joint_idx = int(field_name[-1])  # Extracting joint idx
                self.__fk_joint_targets[joint_idx] = field_value

            else:
                logger.warning(f"Unimplemented QDoubleSpinBox: {field_name}. Value: {field_value}")

        elif isinstance(field_obj, QRadioButton):
            field_value = field_obj.isChecked()

            if field_name == 'Mirror':
                self.__mirror = field_value
                logger.warning("Mirror functionality currently broken")

            else:
                logger.warning(f"Unimplemented QRadioButton: {field_name}. Value: {field_value}")

        else:
            logger.warning(f"Unimplemented QObject: {field_name}. Type: {type(field_obj)}")


class VisualCanvas(FigureCanvasQTAgg):
    """FigureCanvasQTAgg object used to plot the robot state and embed into the Qt interface.

    Attributes:
          fig: Matplotlib Figure.
          axes: Matplotlib Axes.
    """

    def __init__(self, parent=None, width: int = 8, height: int = 8, dpi: int = 100) -> None:
        """Initialise the VisualCanvas object.

        Args:
            parent: Parent of the canvas object.
            width: Width of the canvas object.
            height: Height of the canvas object.
            dpi: Dots per inch of the canvas object.
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.parent = parent
        super().__init__(self.fig)


if __name__ == "__main__":
    app = QApplication([])

    window = ConfigureRobotWindow()

    window.show()

    sys.exit(app.exec())
