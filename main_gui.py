from functools import partial
import sys
import matplotlib
import logging

from helper_functions.helper_functions import MoveType, create_logger
from trajectories import QuinticPolynomialTrajectory

matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QComboBox, QFormLayout, QRadioButton, QVBoxLayout, \
    QPushButton, QMainWindow, QToolBar, QHBoxLayout
from PyQt5.QtCore import QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from robot import Robot
from inverse_kinematics import FabrikSolver
logger = create_logger(module_name=__name__, level=logging.INFO)  # Change debug level as needed


class Window(QMainWindow):

    def __init__(self):
        super().__init__(parent=None)
        self.setWindowTitle("RobotToolbox")
        self._create_central_widget()
        self._create_menu()
        self._create_toolbar()

        self.link_lengths = [0.4, 0.3, 0.2]
        self.ik_solver = FabrikSolver()
        self.joint_configuration = None
        self.robot_base_radius = 0.1
        self.linear_base = True
        self.trajectory_generator = QuinticPolynomialTrajectory()  # TODO add an option on GUI to change this

    def _create_central_widget(self):
        window = QWidget()
        window.setGeometry(100, 100, 280, 80)
        main_window_layout = QVBoxLayout()
        create_robot_button = QPushButton("Create Robot")
        window.setLayout(main_window_layout)
        form_layout = QFormLayout()

        data_fields = {}
        data_fields["link_lengths"] = QLineEdit()
        ik_solver_options = QComboBox()
        ik_solver_options.addItem("--Select an algorithm--")
        ik_solver_options.addItem("Fabrik")
        data_fields["ik_solver"] = ik_solver_options
        data_fields["joint_configuration"] = QLineEdit()
        data_fields["robot_base_radius"] = QLineEdit()
        data_fields["linear_base"] = QRadioButton("On")
        environment = None  # TODO Implement

        form_layout.addRow("Link Lengths:", data_fields["link_lengths"])
        form_layout.addRow("Starting Joint Configuration:", data_fields["joint_configuration"])
        form_layout.addRow("Robot Base Radius:", data_fields["robot_base_radius"])
        form_layout.addRow("Inverse Kinematics Algorithm:", data_fields["ik_solver"])
        form_layout.addRow("Linear Base:", data_fields["linear_base"])

        main_window_layout.addLayout(form_layout)
        main_window_layout.addWidget(create_robot_button)

        self.setCentralWidget(window)
        self._connect_signals(data_fields, create_robot_button)

    def _create_menu(self):
        menu = self.menuBar().addMenu("&Menu")
        menu.addAction("&Exit", self.close)

    def _create_toolbar(self):
        tools = QToolBar()
        tools.addAction("Exit", self.close)
        self.addToolBar(tools)

    def _connect_signals(self, data_fields, button):
        button.clicked.connect(self.launch_robot_control)

        for field_name, field_obj in data_fields.items():
            process_func = partial(self._process_field, field_name, field_obj)
            if field_name == "linear_base":
                field_obj.toggled.connect(process_func)
            elif field_name == "ik_solver":
                field_obj.activated.connect(process_func)
            else:
                field_obj.editingFinished.connect(process_func)

    def _process_field(self, field_id, field_obj):
        # TODO fix excepts
        if field_id in ["link_lengths", "joint_configuration", "robot_base_radius"]:
            field_value = field_obj.text()
            if field_value != "":
                if field_id == "link_lengths":
                    try:
                        self.link_lengths = [float(length) for length in field_value.split(",")]
                    except:
                        raise ValueError("Link lengths must be comma separated numbers")

                elif field_id == "joint_configuration":
                    try:
                        self.joint_configuration = [float(angle) for angle in field_value.split(",")]
                    except:
                        raise ValueError("Joint config must be comma separated numbers")

                elif field_id == "robot_base_radius":
                    try:
                        self.robot_base_radius = float(field_value)
                    except:
                        raise ValueError("Robot base must be a number")

        if field_id == "ik_solver":
            field_value = field_obj.currentText()
            if field_value == "Fabrik":
                self.ik_solver = FabrikSolver()

        if field_id == "linear_base":
            field_value = field_obj.isChecked()
            self.linear_base = field_value

    def launch_robot_control(self):
        robot = Robot(link_lengths=self.link_lengths,
                      ik_solver=self.ik_solver,
                      trajectory_generator=self.trajectory_generator,
                      joint_configuration=self.joint_configuration,
                      robot_base_radius=self.robot_base_radius,
                      linear_base=self.linear_base)
        control_window = ControlWindow(robot)
        control_window.show()


class ControlWindow(QWidget):
    def __init__(self, robot):
        super().__init__()
        self.robot = robot
        self.n_arm_joints = len(self.robot.link_lengths) - 1 if self.robot.linear_base else len(self.robot.link_lengths)
        self.fk_joint_targets = [0.0] * self.n_arm_joints
        self.ik_target_position = [self.robot.vertices['x'][-1], self.robot.vertices['y'][-1]]
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

    def _create_window(self):
        main_layout = QHBoxLayout()
        layout = QVBoxLayout()
        self.setLayout(main_layout)
        self.setWindowTitle("ControlPanel")
        self.setGeometry(750, 750, 750, 750)

        self._create_visual_canvas(main_layout)

        buttons = {}
        fk_form_layout = QFormLayout()
        fk_data_fields = {}
        for joint_idx in range(self.n_arm_joints):
            fk_data_fields[f"Joint {joint_idx}"] = QLineEdit()
            fk_form_layout.addRow(f"Joint {joint_idx}:", fk_data_fields[f"Joint {joint_idx}"])
        buttons["go_fk"] = QPushButton("Go (FK)")

        ik_form_layout = QFormLayout()
        ik_data_fields = {}
        ik_data_fields["Target Position"] = QLineEdit()
        ik_form_layout.addRow("Target Position:", ik_data_fields["Target Position"])
        ik_data_fields["Target Orientation"] = QLineEdit()
        ik_form_layout.addRow("Target Orientation:", ik_data_fields["Target Orientation"])
        ik_data_fields["Mirror"] = QRadioButton()
        ik_form_layout.addRow("Mirror:", ik_data_fields["Mirror"])
        buttons["go_ik"] = QPushButton("Go (IK)")

        layout.addLayout(fk_form_layout)
        layout.addWidget(buttons["go_fk"])
        layout.addStretch(50)
        layout.addLayout(ik_form_layout)
        layout.addWidget(buttons["go_ik"])
        layout.addStretch(250)
        main_layout.addStretch(100)
        main_layout.addLayout(layout)

        self._connect_signals(fk_data_fields, ik_data_fields, buttons)

    def _create_visual_canvas(self, layout):
        self.canvas = VisualCanvas(self)
        layout.addWidget(self.canvas)

        # Allowing click commands to send IK to mouse click
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Start QTimer to update visual canvas
        self.timer.start()

    def on_click(self, event):
        self.ik_target_position = [event.xdata/1000.0, event.ydata/1000.0]
        logger.debug(f"Click event received. Target: {self.ik_target_position}")
        self.get_traj(move_type=MoveType.CARTESIAN, target_position=self.ik_target_position, target_orientation=self.ik_target_orientation)

    # TODO implement click and drag
    def _update_canvas_dynamic(self):
        if len(self.traj) != 0:
            while len(self.traj) > 0:
                config = self.traj.pop(0)
                self.animation_func(config)
            logger.info("Trajectory complete.")
            logger.info(f"Final joint configuration: {self.robot.joint_configuration}")
            logger.info(f"Final vertices: {self.robot.vertices}")
        else:
            self.robot._plot(ax=self.canvas.axes, canvas=self.canvas, mirror=self.mirror)

    def get_traj(self, move_type: MoveType, **kwargs):
        if move_type == MoveType.JOINT:
            self.animation_func = partial(self.robot.move_fk_animated, ax=self.canvas.axes, canvas=self.canvas)
        elif move_type == MoveType.CARTESIAN:
            self.animation_func = partial(self.robot.move_ik_animated, ax=self.canvas.axes, canvas=self.canvas, mirror=False)
        self.traj = self.robot.get_trajectory(move_type, **kwargs)

    def _connect_signals(self, fk_data_fields, ik_data_fields, buttons):
        for joint_name, joint_field_obj in fk_data_fields.items():
            process_func = partial(self._process_field, joint_name, joint_field_obj)
            joint_field_obj.editingFinished.connect(process_func)

        for field_name, field_obj in ik_data_fields.items():
            process_func = partial(self._process_field, field_name, field_obj)
            if field_name == "Mirror":
                field_obj.toggled.connect(process_func)
            else:
                field_obj.editingFinished.connect(process_func)

        buttons["go_fk"].clicked.connect(lambda: self.get_traj(move_type=MoveType.JOINT, target_configuration=self.fk_joint_targets))

        buttons["go_ik"].clicked.connect(lambda: self.get_traj(move_type=MoveType.CARTESIAN, target_position=self.ik_target_position, target_orientation=self.ik_target_orientation))

    def _process_field(self, field_name, field_obj):
        if "Joint" in field_name:
            field_value = field_obj.text()
            if field_value != "":
                try:
                    idx = int(field_name[-1])
                    self.fk_joint_targets[idx] = float(field_value)
                except:
                    raise ValueError("Error for FK")

        elif field_name == 'Target Position':
            field_value = field_obj.text()
            if field_value != "":
                try:
                    self.ik_target_position = [float(coordinate) for coordinate in field_value.split(",")]
                except:
                    raise ValueError('Error for IK target')

        elif field_name == 'Target Orientation':
            field_value = field_obj.text()
            if field_value != "":
                try:
                    self.ik_target_orientation = float(field_value)
                except:
                    raise ValueError('Error for IK target')

        elif field_name == 'Mirror':
            field_value = field_obj.isChecked()
            self.mirror = field_value


class VisualCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)

if __name__ == "__main__":
    app = QApplication([])

    window = Window()

    window.show()

    sys.exit(app.exec())
