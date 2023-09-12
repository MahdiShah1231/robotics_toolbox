import logging
import sys
from functools import partial

import matplotlib
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QComboBox, QFormLayout, QRadioButton, QVBoxLayout, \
    QPushButton, QMainWindow, QToolBar, QHBoxLayout, QBoxLayout, QDoubleSpinBox, QLabel
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


class Window(QMainWindow):

    def __init__(self) -> None:
        super().__init__(parent=None)
        self.setWindowTitle("RobotToolbox")

        self.link_lengths = [0.4, 0.3, 0.2]
        self.ik_solver = FabrikSolver()
        self.joint_configuration = None
        self.robot_base_radius = 0.1
        self.linear_base = True
        self.trajectory_generator = QuinticPolynomialTrajectory()

        self._create_central_widget()
        self._create_menu()
        self._create_toolbar()

    def _create_central_widget(self) -> None:
        window = QWidget()
        window.setGeometry(100, 100, 280, 80)
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
                                                      "for each joint as comma separated values.")
        data_fields["robot_base_radius"].setToolTip("The radius of the robots circular base.")
        data_fields["ik_solver"].setToolTip("The inverse kinematics solver for cartesian space control.")
        data_fields["trajectory_generator"].setToolTip("The trajectory generator for "
                                                       "animated motions between waypoints.")

        # Configuring QWidgets
        data_fields["link_lengths"].setText(','.join(str(val) for val in self.link_lengths))  # Setting defaults
        data_fields["ik_solver"].addItem("Fabrik")
        data_fields["trajectory_generator"].addItem("Quintic Polynomial")
        data_fields["robot_base_radius"].setMinimum(self.robot_base_radius)
        data_fields["robot_base_radius"].setMaximum(1.0)
        data_fields["robot_base_radius"].setSingleStep(0.1)
        data_fields["linear_base"].setChecked(self.linear_base)
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
        menu = self.menuBar().addMenu("&Menu")
        menu.addAction("&Exit", self.close)

    def _create_toolbar(self) -> None:
        tools = QToolBar()
        tools.addAction("Exit", self.close)
        self.addToolBar(tools)

    def _connect_signals(self, data_fields: dict, button: QPushButton) -> None:
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
        # TODO fix excepts
        if isinstance(field_obj, QLineEdit):
            field_value = field_obj.text()

            if field_name == "link_lengths":
                try:
                    self.link_lengths = [float(length) for length in field_value.split(",")]
                except Exception:
                    logger.error("Error processing link lengths", exc_info=True)
                    logger.error("Must be comma separated numbers.")

            elif field_name == "joint_configuration":
                try:
                    self.joint_configuration = [float(angle) for angle in field_value.split(",")]
                except Exception:
                    logger.error("Error processing joint configuration", exc_info=True)
                    logger.error("Must be comma separated numbers.")

            else:
                logger.warning(f"Unimplemented QLineEdit: {field_name}. Value: {field_value}")

        elif isinstance(field_obj, QDoubleSpinBox):
            field_value = field_obj.value()

            if field_name == "robot_base_radius":
                self.robot_base_radius = field_value

            else:
                logger.warning(f"Unimplemented QDoubleSpinBox: {field_name}. Value: {field_value}")

        elif isinstance(field_obj, QComboBox):
            field_value = field_obj.currentText()

            if field_name == "ik_solver":
                self.ik_solver = ik_solvers[field_value]()

            elif field_name == "trajectory_generator":
                self.trajectory_generator = trajectory_generators[field_value]()

            else:
                logger.warning(f"Unimplemented QComboBox: {field_name}. Value: {field_value}")

        elif isinstance(field_obj, QRadioButton):
            field_value = field_obj.isChecked()

            if field_name == "linear_base":
                self.linear_base = field_value

            else:
                logger.warning(f"Unimplemented QRadioButton: {field_name}. Value: {field_value}")

        else:
            logger.warning(f"Unimplemented QObject: {field_name}. Type: {type(field_obj)}")

    def launch_robot_control(self) -> None:
        robot = Robot(link_lengths=self.link_lengths,
                      ik_solver=self.ik_solver,
                      trajectory_generator=self.trajectory_generator,
                      joint_configuration=self.joint_configuration,
                      robot_base_radius=self.robot_base_radius,
                      linear_base=self.linear_base)
        control_window = ControlWindow(robot)
        control_window.show()


class ControlWindow(QWidget):
    def __init__(self, robot) -> None:
        super().__init__()
        self.robot = robot
        self.n_arm_joints = len(self.robot.link_lengths) - 1 if self.robot.linear_base else len(self.robot.link_lengths)
        self.fk_joint_targets = [0.0] * self.n_arm_joints
        self.ik_target_position = [0.0, 0.0]
        self.ik_target_orientation = None
        self.mirror = False
        self.canvas = None

        # Setting up timer to update GUI
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._update_canvas_dynamic)
        self.animation_func = None  # Method to animate trajectory (IK or FK)
        self.traj = []  # List of setpoints to traverse for active trajectory

        self._create_window()

    def _create_window(self) -> None:
        main_layout = QHBoxLayout()
        layout = QVBoxLayout()
        self.setLayout(main_layout)
        self.setWindowTitle("ControlPanel")
        self.setGeometry(750, 750, 750, 750)

        self._create_visual_canvas(main_layout)

        buttons = {
            "go_fk": QPushButton("Go (FK)"),
            "go_ik": QPushButton("Go (IK)")
        }

        fk_form_layout = QFormLayout()
        fk_data_fields = {}
        for joint_idx in range(self.n_arm_joints):
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-3.14, 3.14)
            spinbox.setSingleStep(0.01)
            spinbox.setSuffix(" rad")
            fk_data_fields[f"Joint {joint_idx}"] = spinbox
            fk_form_layout.addRow(f"Joint {joint_idx}:", fk_data_fields[f"Joint {joint_idx}"])

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

        target_position_layout = QHBoxLayout()
        target_position_layout.addWidget(ik_data_fields["Target Position x"])
        target_position_layout.addWidget(ik_data_fields["Target Position y"])

        ik_form_layout.addRow("Target Position (x,y):", target_position_layout)
        ik_form_layout.addRow("Target Orientation:", ik_data_fields["Target Orientation"])
        ik_form_layout.addRow("Mirror:", ik_data_fields["Mirror"])

        layout.addLayout(fk_form_layout)
        layout.addWidget(buttons["go_fk"])
        layout.addStretch(50)
        layout.addLayout(ik_form_layout)
        layout.addWidget(buttons["go_ik"])
        layout.addStretch(250)
        main_layout.addStretch(100)
        main_layout.addLayout(layout)

        self._connect_signals(fk_data_fields, ik_data_fields, buttons)

    def _create_visual_canvas(self, layout: QBoxLayout) -> None:
        self.canvas = VisualCanvas(self)
        layout.addWidget(self.canvas)

        # Allowing click commands to send IK to mouse click
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Start QTimer to update visual canvas
        self.timer.start()

    def on_click(self, event) -> None:
        if event.button == MouseButton.LEFT:
            self.ik_target_position = [event.xdata/1000.0, event.ydata/1000.0]
            logger.debug(f"Click event received. Target: {self.ik_target_position}")
            self.get_traj(move_type=MoveType.CARTESIAN,
                          target_position=self.ik_target_position,
                          target_orientation=self.ik_target_orientation)

        elif event.button == MouseButton.RIGHT:
            logger.warning("Context menu to be implemented")  # TODO implement context menu

    # TODO implement click and drag
    def _update_canvas_dynamic(self) -> None:
        if len(self.traj) != 0:
            while len(self.traj) > 0:
                config = self.traj.pop(0)
                self.animation_func(config)
            logger.info("Trajectory complete.")
            logger.info(f"Final joint configuration: {self.robot.joint_configuration}")
            logger.info(f"Final vertices: {self.robot.vertices}")
        else:
            self.robot._plot(ax=self.canvas.axes, canvas=self.canvas, mirror=self.mirror)

    def get_traj(self, move_type: MoveType, **kwargs) -> None:
        if move_type == MoveType.JOINT:
            self.animation_func = partial(self.robot.move_fk_animated, ax=self.canvas.axes, canvas=self.canvas)
        elif move_type == MoveType.CARTESIAN:
            self.animation_func = partial(self.robot.move_ik_animated, ax=self.canvas.axes, canvas=self.canvas, mirror=False)
        self.traj = self.robot.get_trajectory(move_type, **kwargs)

    def _connect_signals(self, fk_data_fields: dict, ik_data_fields: dict, buttons: dict) -> None:
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

        buttons["go_fk"].clicked.connect(lambda: self.get_traj(move_type=MoveType.JOINT,
                                                               target_configuration=self.fk_joint_targets))

        buttons["go_ik"].clicked.connect(lambda: self.get_traj(move_type=MoveType.CARTESIAN,
                                                               target_position=self.ik_target_position,
                                                               target_orientation=self.ik_target_orientation))

    def _process_field(self, field_name: str, field_obj) -> None:
        if isinstance(field_obj, QDoubleSpinBox):
            field_value = field_obj.value()

            if field_name == 'Target Position x':
                self.ik_target_position[0] = field_value

            elif field_name == 'Target Position y':
                self.ik_target_position[1] = field_value

            elif field_name == 'Target Orientation':
                self.ik_target_orientation = field_value

            elif "Joint" in field_name:  # Field name = "Joint" + idx, for fk target
                joint_idx = int(field_name[-1])  # Extracting joint idx
                self.fk_joint_targets[joint_idx] = field_value

            else:
                logger.warning(f"Unimplemented QDoubleSpinBox: {field_name}. Value: {field_value}")

        elif isinstance(field_obj, QRadioButton):
            field_value = field_obj.isChecked()

            if field_name == 'Mirror':
                self.mirror = field_value
                logger.warning("Mirror functionality currently broken")

            else:
                logger.warning(f"Unimplemented QRadioButton: {field_name}. Value: {field_value}")

        else:
            logger.warning(f"Unimplemented QObject: {field_name}. Type: {type(field_obj)}")


class VisualCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=8, height=4, dpi=100) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


if __name__ == "__main__":
    app = QApplication([])

    window = Window()

    window.show()

    sys.exit(app.exec())
