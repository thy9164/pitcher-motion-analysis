import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import mediapipe as mp

from pitcher_motion_mediapipe import process

btn_font = QFont(None, 12) # font for buttons
window_width = 1920
window_height = 1080
video_player = { # Video player widget parameters
    "position_x": 20,
    "position_y": 20,
    "width": window_width * 3 // 4,
    "height": window_height * 3 // 5,
}

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=2, dpi=100):
        # Create a new figure and one subplot
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.04, right=0.98)
        super(MplCanvas, self).__init__(self.fig)


class CustomSlider(QtWidgets.QSlider):
    """Custom slider class that can paint the contact time indicator"""
    def __init__(self, parent=None):
        super(CustomSlider, self).__init__(parent)
        self.contact_time_position = -1
        self.duration = 1  # Default to avoid division by zero
        self.contact_time_label = None  # Reference to the label
        
    def setContactTimePosition(self, position, duration):
        """Set the contact time position and repaint"""
        self.contact_time_position = position
        self.duration = duration
        self.update()  # Request a repaint
        
        # Update the label position if it exists
        if self.contact_time_label and self.duration > 0:
            contact_time_x_position = int((self.contact_time_position / self.duration) * self.width())
            
            # Center the label under the arrow
            label_width = self.contact_time_label.width()
            new_x = self.mapToParent(QPoint(contact_time_x_position - label_width//2, 0)).x()
            
            # Position the label below the arrow (arrow_y_base + 20 pixels)
            new_y = self.mapToParent(QPoint(0, self.height() + 25)).y()
            
            self.contact_time_label.move(new_x, new_y)
        
    def setContactTimeLabel(self, label):
        """Set the reference to the contact time label"""
        self.contact_time_label = label
        
    def paintEvent(self, event):
        # First, perform the normal slider painting
        super(CustomSlider, self).paintEvent(event)
        
        # Then paint our custom arrow if we have a valid contact time
        if self.contact_time_position >= 0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Set pen and brush for the arrow
            painter.setPen(QPen(Qt.red, 2))
            painter.setBrush(QBrush(Qt.red))
            
            # Calculate the position on the slider corresponding to the contact time
            if self.duration > 0:
                width = self.width()
                contact_time_x_position = int((self.contact_time_position / self.duration) * width)
                
                # Draw a triangle arrow below the slider
                arrow_y_base = self.height() + 5
                
                # Create arrow polygon
                arrow_points = QPolygon([
                    QPoint(contact_time_x_position, arrow_y_base + 15),  # Arrow tip
                    QPoint(contact_time_x_position - 8, arrow_y_base),    # Left corner
                    QPoint(contact_time_x_position + 8, arrow_y_base)     # Right corner
                ])
                
                # Draw filled triangle arrow
                painter.drawPolygon(arrow_points)
                
                # Draw a vertical line through the slider position
                painter.drawLine(
                    contact_time_x_position,
                    0,  # Start from top of slider
                    contact_time_x_position,
                    arrow_y_base  # Down to the arrow
                )

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__() # Initialize the QMainWindow
        self.ui = Ui_MainWindow() # Create an instance of the UI class
        self.ui.setupUi(self) # Setup the UI

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 850)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=False, # it's for video, not static image
                            model_complexity=2, # 0: light, 1: heavy, 2: very heavy
                            min_detection_confidence=0.85, # Confidence threshold for detection
                            min_tracking_confidence=0.85) # Confidence threshold for tracking

        # Video widget to display the video
        self.video_label = QtWidgets.QLabel(self.centralwidget)
        self.video_label.setGeometry(QtCore.QRect(video_player.get("position_x"),
                                                  video_player.get("position_y"),
                                                  video_player.get("width"),
                                                  video_player.get("height")))
        self.video_label.setObjectName("video_label")
        self.video_label.setStyleSheet("background-color: black;")  # Black background
        self.video_label.setAlignment(Qt.AlignCenter)
        
        # Media player to play video
        # self.media_player = QMediaPlayer()
        # self.media_player.setVideoOutput(self.video_widget)

        # Video Controls
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(video_player.get("position_x") + video_player.get("width") + 20,
                                               video_player.get("position_y"),
                                               350, 425))  # Increased height for speed controls
        self.groupBox.setObjectName("groupBox")

        groupBox_button_height = 45

        self.checkbox_left_handed = QtWidgets.QCheckBox("Left Handed", self.groupBox)
        self.checkbox_left_handed.setMinimumHeight(groupBox_button_height)

        self.button_openfile = QtWidgets.QPushButton(self.groupBox)
        self.button_openfile.setMinimumHeight(groupBox_button_height)
        self.button_openfile.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        self.button_openfile.setObjectName("button_openfile")
        self.file_name = None
        self.fps = 0

        # button to detect foot contact
        self.button_process = QtWidgets.QPushButton(self.groupBox)
        self.button_process.setMinimumHeight(groupBox_button_height)
        self.button_process.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        self.button_process.setObjectName("button_process")
        self.button_process.setText("Process")
        self.button_process.setEnabled(False)

        self.foot_contact_frame_num = 0
        self.foot_velocity_xs = []
        self.foot_velocity_ys = []
        self.frame_nums = []
        self.max_external_rotation_frame_num = 0
        self.max_external_rotation_angle = 0
        self.right_arm_angles = []
        self.release_frame_num = 0
        self.right_arm_angle_difference = []
        self.leg_angles = []
        self.center_mass_xs = []
        self.center_mass_ys = []

        self.button_contact = QtWidgets.QPushButton(self.groupBox)
        self.button_contact.setMinimumHeight(groupBox_button_height)
        self.button_contact.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        self.button_contact.setObjectName("button_contact")
        self.button_contact.setText("Foot Contact")
        self.button_contact.setEnabled(False)

        self.button_max_external_rotation = QtWidgets.QPushButton(self.groupBox)
        self.button_max_external_rotation.setMinimumHeight(groupBox_button_height)
        self.button_max_external_rotation.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        self.button_max_external_rotation.setObjectName("button_max_external_rotation")
        self.button_max_external_rotation.setText("Max External Rotation")
        self.button_max_external_rotation.setEnabled(False)

        self.button_release = QtWidgets.QPushButton(self.groupBox)
        self.button_release.setMinimumHeight(groupBox_button_height)
        self.button_release.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        self.button_release.setObjectName("button_release")
        self.button_release.setText("Ball Release")
        self.button_release.setEnabled(False)

        self.button_plot_leg_angle = QtWidgets.QPushButton(self.groupBox)
        self.button_plot_leg_angle.setMinimumHeight(groupBox_button_height)
        self.button_plot_leg_angle.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        self.button_plot_leg_angle.setObjectName("button_plot_leg_angle")
        self.button_plot_leg_angle.setText("Plot Knee Angles")
        self.button_plot_leg_angle.setEnabled(False)

        self.button_clean_canvas = QtWidgets.QPushButton(self.groupBox)
        self.button_clean_canvas.setMinimumHeight(groupBox_button_height)
        self.button_clean_canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        self.button_clean_canvas.setObjectName("button_clean_canvas")
        self.button_clean_canvas.setText("Clean Canvas")
        self.button_clean_canvas.setEnabled(False)

        self.file_buttons_layout = QVBoxLayout(self.groupBox)
        self.file_buttons_layout.setContentsMargins(15, 15, 15, 15)
        self.file_buttons_layout.setSpacing(8)

        self.file_buttons_layout.addWidget(self.checkbox_left_handed)
        self.file_buttons_layout.addWidget(self.button_openfile)
        self.file_buttons_layout.addWidget(self.button_process)
        self.file_buttons_layout.addWidget(self.button_contact)
        self.file_buttons_layout.addWidget(self.button_max_external_rotation)
        self.file_buttons_layout.addWidget(self.button_release)
        self.file_buttons_layout.addWidget(self.button_plot_leg_angle)
        self.file_buttons_layout.addWidget(self.button_clean_canvas)

        self.file_buttons_layout.addStretch(1)

        self.dataBox = QtWidgets.QGroupBox(self.centralwidget)
        self.dataBox.setGeometry(QtCore.QRect(self.groupBox.x(),
                                              self.groupBox.y() + self.groupBox.height() + 10,
                                              350, 100))
        
        self.label_knee_angle = QtWidgets.QLabel(self.dataBox)
        self.label_knee_angle.setFixedHeight(groupBox_button_height)
        self.label_knee_angle.setObjectName("label_knee_angle")
        self.label_knee_angle.setVisible(False)
        self.total_knee_extension = 0

        self.label_MER = QtWidgets.QLabel(self.dataBox)
        self.label_MER.setFixedHeight(groupBox_button_height)
        self.label_MER.setObjectName("label_MER")
        self.label_MER.setVisible(False)
        self.total_knee_extension = 0

        self.dataBox_layout = QVBoxLayout(self.dataBox)
        self.dataBox_layout.addWidget(self.label_knee_angle)
        self.dataBox_layout.addWidget(self.label_MER)
        
        # Timer for video playback
        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.update_frame)
        
        self.playback_clock = QElapsedTimer()
        self.playback_start_frame = 0
        self.playback_speed = 1.0

        # OpenCV Video Capture Object
        self.cap = None
        self.is_paused = True

        # Use our custom slider for video progress
        self.slider_videoframe = CustomSlider(self.centralwidget)
        self.slider_videoframe.setGeometry(QtCore.QRect(video_player.get("position_x"),
                                                        video_player.get("position_y") + video_player.get("height") + 10,
                                                        video_player.get("width") - 120,
                                                        40))
        self.slider_videoframe.setOrientation(QtCore.Qt.Horizontal)
        self.slider_videoframe.setObjectName("slider_videoframe")

        # Label to display the current video frame
        self.label_videoframe = QtWidgets.QLabel(self.centralwidget)
        self.label_videoframe.setGeometry(QtCore.QRect(self.slider_videoframe.x() + self.slider_videoframe.width() + 10,
                                                       self.slider_videoframe.y(),
                                                       120, 40))
        self.label_videoframe.setObjectName("label_videoframe")
        self.label_videoframe.setText("Frame: 0/0")
        self.current_frame = 0  # Initialize current frame
        self.total_frames = 0  # Initialize total frames


        self.video_controll_box = QtWidgets.QGroupBox(self.centralwidget)
        self.video_controll_box.setGeometry(QtCore.QRect(self.groupBox.x(),
                                                         self.slider_videoframe.y() + self.slider_videoframe.height() - 150,
                                                         350, 150))  # Increased height for speed controls
        self.video_controll_box.setObjectName("video_controll_box")

        self.button_play = QtWidgets.QPushButton(self.video_controll_box)
        self.button_play.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed
        )
        self.button_play.setObjectName("button_play")
        self.button_play.setText("Play")  # Replacing icon with text

        self.button_pause = QtWidgets.QPushButton(self.video_controll_box)
        self.button_pause.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed
        )
        self.button_pause.setObjectName("button_pause")
        self.button_pause.setText("Pause")  # Replacing icon with text

        self.button_previous_frame = QtWidgets.QPushButton(self.video_controll_box)
        self.button_previous_frame.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed
        )
        self.button_previous_frame.setObjectName("button_previous_frame")
        self.button_previous_frame.setText("Prev")  # Replacing icon with text

        self.button_next_frame = QtWidgets.QPushButton(self.video_controll_box)
        self.button_next_frame.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed
        )
        self.button_next_frame.setObjectName("button_next_frame")
        self.button_next_frame.setText("Next")  # Replacing icon with text

        # Playback Speed Control: Add a slider for playback speed
        self.slider_speed = QtWidgets.QSlider(self.video_controll_box)
        # self.slider_speed.setGeometry(QtCore.QRect(10, 250, 100, 30))
        self.slider_speed.setMinimumWidth(80)
        self.slider_speed.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed
        )
        self.slider_speed.setOrientation(QtCore.Qt.Horizontal)
        self.slider_speed.setRange(10, 200)  # 10% to 200% speed
        self.slider_speed.setValue(100)  # Default speed is 100%
        self.slider_speed.setObjectName("slider_speed")

        self.checkbox_repeat = QtWidgets.QCheckBox("Repeat", self.video_controll_box)

        # Label to display the playback speed
        self.label_speed = QtWidgets.QLabel(self.video_controll_box)
        self.label_speed.setObjectName("label_speed")
        self.label_speed.setText("Speed: 100%")  # Default to 100%

        self.controll_hbox = QHBoxLayout() # Make the horizontal layout for the bottom row
        self.controll_hbox.addWidget(self.button_previous_frame)
        self.controll_hbox.addSpacing(10)
        self.controll_hbox.addWidget(self.button_pause)
        self.controll_hbox.addSpacing(10)
        self.controll_hbox.addWidget(self.button_next_frame)
        self.controll_hbox.setAlignment(Qt.AlignHCenter)

        self.controll_hbox2 = QHBoxLayout() # Make the horizontal layout for the bottom row
        self.controll_hbox2.addWidget(self.slider_speed)
        self.controll_hbox2.addSpacing(10)
        self.controll_hbox2.addWidget(self.label_speed)
        self.controll_hbox2.setAlignment(Qt.AlignHCenter)

        self.controll_box_layout = QVBoxLayout(self.video_controll_box)
        self.controll_box_layout.addWidget(self.button_play, alignment=Qt.AlignHCenter)
        self.controll_box_layout.addLayout(self.controll_hbox)
        self.controll_box_layout.addLayout(self.controll_hbox2)
        self.controll_box_layout.addWidget(self.checkbox_repeat)
        
        # Create an instance of the Matplotlib canvas
        self.canvas = MplCanvas(self.centralwidget, width=1, height=1, dpi=100)
        self.canvas.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        # Create a container QFrame for the plot layout
        self.plot_frame = QtWidgets.QFrame(self.centralwidget)
        self.plot_frame.setGeometry(QtCore.QRect(self.slider_videoframe.x(), 
                                                self.slider_videoframe.y() + self.slider_videoframe.height(), 
                                                MainWindow.width() - 100, 
                                                MainWindow.height() - (self.slider_videoframe.y() + self.slider_videoframe.height() + 120)))

        # Create a layout and add it to the frame
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_frame)
        # Add your canvas or other widgets to the layout
        self.plot_layout.addWidget(self.canvas)
        self.plot_status = "example"
        self.plot_xline = None
        
        self.plot_example()

        # =========================
        # Responsive layout
        # =========================

        # 整個視窗：上下排列
        self.main_layout = QVBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(12, 12, 12, 12)
        self.main_layout.setSpacing(10)

        # -------------------------
        # 左側：影片 + timeline
        # -------------------------
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(8)

        # 影片會自動吃掉可以使用的空間
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        self.video_label.setMinimumSize(480, 270)

        self.left_layout.addWidget(self.video_label, 1)

        # slider 和 frame number 放在同一排
        self.timeline_layout = QHBoxLayout()

        self.slider_videoframe.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed
        )

        self.timeline_layout.addWidget(self.slider_videoframe, 1)
        self.timeline_layout.addWidget(self.label_videoframe)

        self.left_layout.addLayout(self.timeline_layout)

        # -------------------------
        # 右側：所有控制元件
        # -------------------------
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(8)

        self.right_panel.setMinimumWidth(300)
        self.right_panel.setMaximumWidth(420)

        self.right_layout.addWidget(self.groupBox)
        self.right_layout.addWidget(self.dataBox)
        self.right_layout.addWidget(self.video_controll_box)
        self.right_layout.addStretch(1)

        # -------------------------
        # 上半部：左邊影片 + 右邊控制區
        # -------------------------
        self.top_layout = QHBoxLayout()
        self.top_layout.setSpacing(10)

        self.top_layout.addWidget(self.left_panel, 4)
        self.top_layout.addWidget(self.right_panel, 1)

        # -------------------------
        # 整體：上半部 + 下方 plot
        # -------------------------
        self.main_layout.addLayout(self.top_layout, 3)
        self.main_layout.addWidget(self.plot_frame, 1)


        # Connecting buttons to functions
        self.button_openfile.clicked.connect(self.open_file_clicked)
        self.button_play.clicked.connect(self.play_video)
        self.button_pause.clicked.connect(self.pause_video)
        self.button_process.clicked.connect(self.process)
        self.button_plot_leg_angle.clicked.connect(self.plot_leg_angle)
        self.button_clean_canvas.clicked.connect(self.clean_canvas)
        self.slider_speed.valueChanged.connect(self.adjust_playback_speed)  # Connect slider to speed function
        self.slider_videoframe.sliderMoved.connect(self.seek_video)
        self.slider_videoframe.actionTriggered.connect(self.slider_action)
        self.button_previous_frame.clicked.connect(self.previous_frame)
        self.button_next_frame.clicked.connect(self.next_frame)
        self.button_contact.clicked.connect(self.contact_frame)
        self.button_max_external_rotation.clicked.connect(self.max_ER_frame)
        self.button_release.clicked.connect(self.release_frame)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def plot_example(self):
        self.canvas.draw()

    def plot_frame_xline(self, x_line):
        if self.frame_nums[1] <= x_line and x_line < self.frame_nums[-1] and self.file_name == "output.mp4":
            label = "None"
            if self.plot_xline is not None:
                if self.plot_xline in self.canvas.ax.lines:
                    self.plot_xline.remove()
                self.plot_xline = None
            if self.plot_status == "arm angle" and self.right_arm_angles[x_line-1] != None:
                label = f"angle: {round(self.right_arm_angles[x_line-1])}"
            elif self.plot_status == "leg angle" and self.leg_angles[x_line-1] != None:
                label = f"angle: {round(self.leg_angles[x_line-1])}"
            elif self.plot_status == "center mass" and self.center_mass_xs[x_line-1] != None:
                label = f"{round(self.center_mass_xs[x_line-1])}, {round(self.center_mass_ys[x_line-1])}"
            self.plot_xline = self.canvas.ax.axvline(x=x_line, linewidth=3, color='gray', linestyle='--', label=label)
            self.canvas.ax.legend()
            self.canvas.draw()
    
    def plot (self, option):
        if option == "foot_velo":
            self.plot_foot_contact(x_line=self.current_frame)
        elif option == "arm_angle":
            self.plot_right_arm_angle(x_line=self.current_frame)
        elif option == "leg_angle":
            self.plot_leg_angle()
    
    def plot_foot_contact(self, x_line):
        # Plot foot contact detection results
        self.plot_status = "foot velo"
        if self.foot_contact_frame_num >= 0 and len(self.frame_nums) > 0 and self.file_name == "output.mp4":
            self.canvas.ax.clear()
            self.canvas.ax.plot(self.frame_nums, self.foot_velocity_xs, label='Foot Velocity X', color='blue')
            self.canvas.ax.plot(self.frame_nums, self.foot_velocity_ys, label='Foot Velocity Y', color='red')
            self.canvas.ax.axhline(y=0, color='black')
            self.canvas.ax.set_title("Foot contact Detection")
            self.canvas.ax.set_xlabel("Time (Frame Number)")
            self.canvas.ax.set_ylabel("Velocity")
            if self.frame_nums[1] <= x_line and x_line <= self.frame_nums[-1]:
                self.canvas.ax.axvline(x=x_line-0.5, color='green', linestyle='--', label=f"x: {self.foot_velocity_xs[x_line-1]}, y: {self.foot_velocity_ys[x_line-1]}")
                self.canvas.ax.set_xlim(x_line - 50 if x_line > 50 else 0, x_line + 50 if x_line + 50 < self.frame_nums[-1] else self.frame_nums[-1])  # Extend x-axis limit for better visibility
            self.canvas.ax.grid(True)
            self.canvas.ax.legend() # Add legend to the plot

            # Force x-axis ticks to show integer only values
            import matplotlib.ticker as ticker
            self.canvas.ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            # Adjust bottom margin to make space for xlabel and tick labels
            self.canvas.fig.tight_layout(pad=1.0)

            self.canvas.draw()

    def plot_right_arm_angle(self, x_line):
        self.plot_status = "arm angle"
        if self.foot_contact_frame_num >= 0 and len(self.right_arm_angles) > 0 and self.file_name == "output.mp4":
            self.canvas.ax.clear()
            self.canvas.ax.plot(self.frame_nums, self.right_arm_angles, label='Angle', color='blue')
            self.canvas.ax.axhline(y=0, color='black')
            self.canvas.ax.set_title("Arm Angles")
            self.canvas.ax.set_xlabel("Time (Frame Number)")
            self.canvas.ax.set_ylabel("Angle (degree)")
            if self.frame_nums[1] <= x_line and x_line <= self.frame_nums[-1]: 
                self.canvas.ax.axvline(x=x_line, color='green', linestyle='--', label=f"angle: {self.right_arm_angles[x_line-1]}")
                self.canvas.ax.axvline(x=self.foot_contact_frame_num, color='black', label=f"foot_contact")
                self.canvas.ax.axvline(x=self.release_frame_num, color='black', label=f"release")
                self.canvas.ax.set_xlim(self.foot_contact_frame_num, self.release_frame_num)  # Extend x-axis limit for better visibility
            self.canvas.ax.grid(True)
            self.canvas.ax.legend() # Add legend to the plot

            # Force x-axis ticks to show integer only values
            import matplotlib.ticker as ticker
            self.canvas.ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            # Adjust bottom margin to make space for xlabel and tick labels
            self.canvas.fig.tight_layout(pad=1.0)

            self.canvas.draw()

    def clean_canvas(self):
        self.canvas.ax.clear()
        self.canvas.draw()
    
    def plot_data(self, title, ylabel, label1, data1, label2="", data2=None):
        if len(data1) > 0 and len(data1) == len(self.frame_nums) and self.file_name == "output.mp4":
            self.canvas.ax.clear()
            self.canvas.ax.plot(self.frame_nums, data1, label=label1, linewidth=2, color='blue')
            if data2 != None and len(data2) == len(self.frame_nums):
                self.canvas.ax.plot(self.frame_nums, data2, label=label2, color='red')
            # self.canvas.ax.axhline(y=0, color='black')
            self.canvas.ax.set_title(title)
            self.canvas.ax.set_xlabel("Time (Frame Number)")
            self.canvas.ax.set_ylabel(ylabel)
            if self.current_frame > 0:
                self.plot_frame_xline(self.current_frame)
            if self.foot_contact_frame_num > 0:
                self.canvas.ax.axvline(x=self.foot_contact_frame_num, linewidth=3, color='red', label=f"foot contact")
            if self.max_external_rotation_frame_num > 0:
                self.canvas.ax.axvline(x=self.max_external_rotation_frame_num, linewidth=3, color='orange', label="max external rotation")
            if self.release_frame_num > 0:
                self.canvas.ax.axvline(x=self.release_frame_num, linewidth=3, color='green', label=f"release")
            if self.foot_contact_frame_num > 0 and self.release_frame_num > self.foot_contact_frame_num:
                arm_phases_frame_nums = self.frame_nums[self.release_frame_num] - self.frame_nums[self.foot_contact_frame_num]
                self.canvas.ax.set_xlim(self.foot_contact_frame_num - arm_phases_frame_nums if self.foot_contact_frame_num > arm_phases_frame_nums else 0, self.release_frame_num + arm_phases_frame_nums if self.release_frame_num + arm_phases_frame_nums < self.frame_nums[-1] else self.frame_nums[-1])  # Extend x-axis limit for better visibility
            if self.plot_status == "leg angle" and self.leg_angles[self.foot_contact_frame_num-5:self.release_frame_num+5] and all(angle is not None for angle in self.leg_angles[self.foot_contact_frame_num-5:self.release_frame_num+5]):
                self.canvas.ax.grid(True)
                self.canvas.ax.set_ylim(min(self.leg_angles[self.foot_contact_frame_num-3:self.release_frame_num+3])-5, 
                                        max(self.leg_angles[self.foot_contact_frame_num-3:self.release_frame_num+3])+5)
                print("knee angle", min(self.leg_angles[self.foot_contact_frame_num-5:self.release_frame_num+5])-10, 
                    max(self.leg_angles[self.foot_contact_frame_num-5:self.release_frame_num+5])+10)
            self.canvas.ax.legend(fontsize=13) # Add legend to the plot

            # Force x-axis ticks to show integer only values
            import matplotlib.ticker as ticker
            self.canvas.ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            # Adjust bottom margin to make space for xlabel and tick labels
            self.canvas.fig.tight_layout(pad=1.0)

            self.canvas.draw()
            self.canvas.fig.savefig('plot.png')

    def plot_leg_angle(self):
        self.plot_status = "leg angle"
        for i in range(len(self.leg_angles)):
            if self.leg_angles[i] is not None and self.leg_angles[i-1] is not None:
                if abs(self.leg_angles[i] - self.leg_angles[i-1]) > 50:
                    self.leg_angles[i] = self.leg_angles[i-1]
        self.plot_data("Leg Angle", "Angle (degree)", 'Angle', self.leg_angles)
        # if len(self.leg_angles) > 0 and len(self.leg_angles) == len(self.frame_nums) and self.file_name == "output.mp4":
        #     self.canvas.ax.clear()
        #     self.canvas.ax.plot(self.frame_nums, self.leg_angles, label='Angle', color='blue')
        #     self.canvas.ax.axhline(y=0, color='black')
        #     self.canvas.ax.set_title("Leg Angles")
        #     self.canvas.ax.set_xlabel("Time (Frame Number)")
        #     self.canvas.ax.set_ylabel("Angle (degree)")
        #     self.plot_frame_xline(self.current_frame)
        #     if self.foot_contact_frame_num > 0:
        #         self.canvas.ax.axvline(x=self.foot_contact_frame_num, linewidth=1, color='red', label=f"foot contact")
        #     if self.max_external_rotation_frame_num > 0:
        #         self.canvas.ax.axvline(x=self.max_external_rotation_frame_num, linewidth=1, color='orange', label="max external rotation")
        #     if self.release_frame_num > 0:
        #         self.canvas.ax.axvline(x=self.release_frame_num, linewidth=1, color='black', label=f"release")
        #     if self.foot_contact_frame_num > 0 and self.release_frame_num > self.foot_contact_frame_num:
        #         self.canvas.ax.set_xlim(self.foot_contact_frame_num - 20 if self.foot_contact_frame_num > 20 else 0, self.release_frame_num + 50 if self.release_frame_num + 50 < self.frame_nums[-1] else self.frame_nums[-1])  # Extend x-axis limit for better visibility
        #     # self.canvas.ax.grid(True)
        #     self.canvas.ax.legend() # Add legend to the plot

        #     # Force x-axis ticks to show integer only values
        #     import matplotlib.ticker as ticker
        #     self.canvas.ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
        #     # Adjust bottom margin to make space for xlabel and tick labels
        #     self.canvas.fig.tight_layout(pad=1.0)

        #     self.canvas.draw()

    def plot_center_mass(self):
        self.plot_status = "center mass"
        self.plot_data("Center of Mass", "center of mass (pixel)", 'x', self.center_mass_xs, 'y', self.center_mass_ys)

    def open_file(self):
        if self.file_name:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release() 
            self.button_process.setEnabled(True)
            self.button_contact.setEnabled(False)
            self.button_max_external_rotation.setEnabled(False)
            self.button_release.setEnabled(False)
            self.button_plot_leg_angle.setEnabled(False)
            self.button_clean_canvas.setEnabled(False)

            self.clean_canvas()

            self.cap = cv2.VideoCapture(self.file_name)
            if not self.cap.isOpened():
                QMessageBox.critical(self.centralwidget, "Error", "Could not open video file.")
                return
            print(f"Opened video file: {self.file_name}")

    def open_file_clicked(self):
        # Open file dialog to select a video file
        self.file_name, _ = QFileDialog.getOpenFileName(self.centralwidget, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)")
        self.open_file()
        self.checkbox_left_handed.setChecked(False)
        self.seek_video(0, update_slider=True)  # Seek to the beginning of the video
        self.play_video()

    def play_video(self):
        if not self.file_name:
            QMessageBox.critical(
                self.centralwidget,
                "Error",
                "No video file opened."
            )
            return
    
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.file_name)
    
        if not self.cap.isOpened():
            QMessageBox.critical(
                self.centralwidget,
                "Error",
                "Could not open video file."
            )
            return
    
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    
        if self.total_frames <= 0:
            QMessageBox.critical(
                self.centralwidget,
                "Error",
                "Could not determine total frames."
            )
            return
    
        if self.fps <= 0:
            QMessageBox.critical(
                self.centralwidget,
                "Error",
                "Could not determine video FPS."
            )
            return
    
        current_position = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
        # 如果已經播完，從頭開始
        if current_position >= self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_position = 0
    
        self.playback_speed = self.slider_speed.value() / 100.0
    
        # 記錄「開始播放時在哪一格」
        self.playback_start_frame = current_position
    
        # 真實時間從這裡開始計時
        self.playback_clock.restart()
    
        self.is_paused = False
        self.timer.start()

    def pause_video(self):
        self.is_paused = True
        self.timer.stop()
    
        if (
            self.foot_contact_frame_num > 0
            and len(self.frame_nums) > 1
        ):
            self.plot_frame_xline(
                self.current_frame
            )

    def frame_processing(self, frame):
        # Convert the frame from BGR to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
            
        # Get the dimensions of the video label
        label_width = self.video_label.width()
        label_height = self.video_label.height()

        # Compute new dimensions while keeping aspect ratio
        aspect_ratio = w / h
        if label_width / label_height > aspect_ratio:
            # Height is the limiting factor
            new_height = label_height
            new_width = int(aspect_ratio * new_height)
        else:
            new_width = label_width
            new_height = int(new_width / aspect_ratio)

        # Convert to QPixmap and set it on the label
        pixmap = QPixmap.fromImage(QImage(rgb_image, w, h, (ch * w), QImage.Format_RGB888))
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
        ))

    def update_frame(self):
        if (
            self.cap is None
            or not self.cap.isOpened()
            or self.is_paused
            or self.fps <= 0
        ):
            return
    
        elapsed_seconds = self.playback_clock.elapsed() / 1000.0
    
        frames_elapsed = int(
            elapsed_seconds
            * self.fps
            * self.playback_speed
        )
    
        target_position = (
            self.playback_start_frame
            + frames_elapsed
        )
    
        reached_end = False
    
        if target_position >= self.total_frames:
            target_position = self.total_frames
            reached_end = True
    
        current_position = int(
            self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        )
    
        # 還沒到下一格的時間
        if target_position <= current_position:
            return
    
        # 如果程式落後，就跳過中間不用顯示的 frame
        grabbed = False
    
        while current_position < target_position:
            if not self.cap.grab():
                reached_end = True
                break
    
            current_position += 1
            grabbed = True
    
        # 只顯示最後應該看到的那一格
        if grabbed:
            ret, frame = self.cap.retrieve()
    
            if ret:
                self.frame_processing(frame)
    
                # 播放時不要每一格重畫 Matplotlib
                self.update_slider_position(
                    update_plot=False
                )
    
        if reached_end:
            if self.checkbox_repeat.isChecked():
                self.seek_video(
                    0,
                    frame_to_seek=1,
                    update_slider=True
                )
    
                self.playback_start_frame = int(
                    self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                )
    
                self.playback_clock.restart()
    
            else:
                self.timer.stop()
                self.is_paused = True
    
                # 播完後再更新一次 plot
                if (
                    self.foot_contact_frame_num > 0
                    and len(self.frame_nums) > 1
                ):
                    self.plot_frame_xline(
                        self.current_frame
                    )    
    
    def process(self):
        self.pause_video()  # Pause the video before detection
        if self.file_name is None:
            QMessageBox.critical(self.centralwidget, "Error", "No video file opened.")
            return
        else:
            self.button_openfile.setEnabled(False)
            self.button_process.setEnabled(False)
            self.canvas.ax.clear()  # ax 是 axes 的縮寫
            self.canvas.draw()
            (
                self.foot_contact_frame_num, self.foot_velocity_xs, self.foot_velocity_ys, self.frame_nums,
                self.max_external_rotation_frame_num, self.max_external_rotation_angle, self.right_arm_angles,
                self.release_frame_num, self.right_arm_angle_difference,
                self.leg_angles, self.center_mass_xs, self.center_mass_ys
            ) = 0, [], [], [], 0, 0, [], 0, [], [], [], []
            (
                self.foot_contact_frame_num, self.foot_velocity_xs, self.foot_velocity_ys, self.frame_nums,
                self.max_external_rotation_frame_num, self.max_external_rotation_angle, self.right_arm_angles,
                self.release_frame_num, self.right_arm_angle_difference,
                self.leg_angles, self.center_mass_xs, self.center_mass_ys
            ) = process(self.file_name, left_handed=self.checkbox_left_handed.isChecked())
            self.button_openfile.setEnabled(True)
            
            if self.foot_contact_frame_num > 0 and self.max_external_rotation_frame_num > 0 and self.release_frame_num > 0:
                self.file_name = "output.mp4"
                self.open_file()

                if not self.cap.isOpened():
                    QMessageBox.critical(self.centralwidget, "Error", "Could not open video file.")
                    return

                else:
                    self.button_contact.setEnabled(True)
                    self.button_max_external_rotation.setEnabled(True)
                    self.button_release.setEnabled(True)
                    self.button_plot_leg_angle.setEnabled(True)
                    self.button_clean_canvas.setEnabled(True)

                    
                    # if self.max_external_rotation_angle > 0:
                    #     self.label_MER.setText(f"Max External Rotation = {round(self.max_external_rotation_angle)}°")
                    #     self.label_MER.setVisible(True)

                    # total knee extension = knee flexion angle at foot contact - knee flexion angle at release
                    if self.leg_angles[self.foot_contact_frame_num-1] is not None and self.leg_angles[self.release_frame_num-1] is not None:
                        self.total_knee_extension = round(self.leg_angles[self.foot_contact_frame_num-1] - self.leg_angles[self.release_frame_num-1])
                        self.label_knee_angle.setText(f"Total Knee Extension = {self.total_knee_extension}°")
                        self.label_knee_angle.setVisible(True)

                    self.plot_leg_angle()  # Plot the foot contact detection results
                    self.plot_status = "leg angle"
                    print(f"frame {self.foot_contact_frame_num}: Foot contact")
                    print(f"frame {self.max_external_rotation_frame_num}: max_external_rotation_angle: {self.max_external_rotation_angle}")
                    print(f"frame {self.release_frame_num}: release")
                    self.button_process.setEnabled(False)  # Disable the button after detection
                    self.seek_video(0, frame_to_seek=self.release_frame_num, update_slider=True)  # Seek to the detected frame
            else:
                QMessageBox.critical(self.centralwidget, "Process Error", "Something went wrong during the processing.\nPlease try again.")
                self.button_process.setEnabled(True)

    def update_slider_position(self, update_plot=True):
        # Update the slider based on the current position of the video
        if self.cap is not None and self.cap.isOpened():
            # Get current frame index and total frame count
            if self.total_frames == 0:
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.is_paused == False:
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.label_videoframe.setText(f"Frame: {self.current_frame}/{self.total_frames}")
            if self.total_frames > 0:
                # Calculate the percentage position and set the slider value
                progress = int((self.current_frame / self.total_frames) * 100)
                self.slider_videoframe.setValue(progress)
                if update_plot and self.foot_contact_frame_num > 0:
                    self.plot_frame_xline(self.current_frame)         

    def slider_action(self, action):
        self.seek_video(position=self.slider_videoframe.sliderPosition())

    def seek_video(self, position, frame_to_seek=0, update_slider=False):
        # Seek to the selected position in the video
        if self.cap is not None and self.cap.isOpened():
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.total_frames > 0:
                # Calculate the frame to seek to
                if frame_to_seek == 0:
                    frame_to_seek = int((position / 100) * self.total_frames)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_seek-1)  # Set to the frame before the desired one
                self.current_frame = frame_to_seek
                self.label_videoframe.setText(f"Frame: {self.current_frame}/{self.total_frames}")
                if update_slider:
                    self.update_slider_position()
                ret, frame = self.cap.read()
                if ret:
                    self.frame_processing(frame)

                # 如果播放途中 seek，從新的位置重新同步播放時鐘
                if not self.is_paused:
                    self.playback_start_frame = int(
                        self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    )
                    self.playback_clock.restart()

    def previous_frame(self):
        # Move back one frame if possible
        if self.current_frame > 0:
            self.current_frame -= 1
            self.seek_video(0, frame_to_seek=self.current_frame, update_slider=True)  # Seek to the previous frame

    def next_frame(self):
        # Move forward one frame if possible
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.seek_video(0, frame_to_seek=self.current_frame, update_slider=True)  # Seek to the next frame

    def contact_frame(self):
        if self.foot_contact_frame_num > 0:
            self.seek_video(0, self.foot_contact_frame_num, update_slider=True)

    def max_ER_frame(self):
        if self.max_external_rotation_frame_num > 0:
            self.seek_video(0, self.max_external_rotation_frame_num, update_slider=True)

    def release_frame(self):
        if self.release_frame_num > 0:
            self.seek_video(0, self.release_frame_num, update_slider=True)

    def adjust_playback_speed(self):
        new_speed = self.slider_speed.value() / 100.0
    
        self.label_speed.setText(
            f"Speed: {self.slider_speed.value()}%"
        )
    
        # 如果正在播放，改速度時從目前 frame 重新開始計時
        if (
            self.timer.isActive()
            and not self.is_paused
            and self.cap is not None
            and self.cap.isOpened()
        ):
            self.playback_start_frame = int(
                self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            )
    
            self.playback_clock.restart()
    
        self.playback_speed = new_speed

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pitcher Motion Segmentation and Analysis"))
        
        self.button_openfile.setText(_translate("MainWindow", "Open Video"))
        self.button_play.setText(_translate("MainWindow", "Play"))
        self.button_pause.setText(_translate("MainWindow", "Pause"))

        self.checkbox_left_handed.setFont(btn_font)
        self.button_openfile.setFont(btn_font)
        self.button_process.setFont(btn_font)
        self.button_contact.setFont(btn_font)
        self.button_max_external_rotation.setFont(btn_font)
        self.button_release.setFont(btn_font)

        self.button_plot_leg_angle.setFont(btn_font)
        self.button_clean_canvas.setFont(btn_font)
        self.label_knee_angle.setFont(btn_font)
        self.label_MER.setFont(btn_font)

        self.button_play.setFont(btn_font)
        self.button_pause.setFont(btn_font)
        self.button_previous_frame.setFont(btn_font)
        self.button_next_frame.setFont(btn_font)
        self.label_speed.setFont(QFont(None, 10))
        self.checkbox_repeat.setFont(QFont(None, 10))
        self.label_videoframe.setFont(QFont(None, 9))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.showMaximized()
    sys.exit(app.exec_())