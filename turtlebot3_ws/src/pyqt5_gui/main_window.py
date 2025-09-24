import sys
import os
import time
import threading
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QTextEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QProgressBar, QDial
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QTransform
from PyQt5.QtCore import Qt, QTimer, QSize

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist, Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from ros2_image_node import ImageROS2Interface
from std_msgs.msg import Int32
import time


class TrafficSignListener(Node):
    def __init__(self, log_callback):
        super().__init__('traffic_sign_listener_gui')
        self.log_callback = log_callback
        self.create_subscription(Int32, '/detect/traffic_sign', self.sign_callback, 10)

    def sign_callback(self, msg):
        now = time.time()
        if now - self.last_log_time < 1.0:
            return  # 1초 이내면 로그 무시
        self.last_log_time = now
        log_messages = {
            1: "주차장을 감지하였습니다.",
            # 2: "멈춥니다.",
            3: "왼쪽으로 꺾습니다.",
            4: "오른쪽으로 꺾습니다.",
            5: "빨간불입니다.",
            6: "노란불입니다.",
            7: "초록불입니다."
        }
        message = log_messages.get(msg.data, f"알 수 없는 신호: {msg.data}")
        self.log_callback(f"[GAZEBO] {message}")

class SignTogglePublisher(Node):
    def __init__(self):
        super().__init__('sign_toggle_gui')
        self.publisher_ = self.create_publisher(Bool, '/sign_toggle', 10)

    def toggle(self, state: bool):
        msg = Bool()
        msg.data = state
        self.publisher_.publish(msg)

class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher_gui')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self._running = False 

    def publish_continuous(self, linear_x=0.0, angular_z=0.0):
        self._running = True
        def loop():
            while self._running:
                twist = Twist()
                twist.linear.x = linear_x
                twist.angular.z = angular_z
                self.publisher_.publish(twist)
                time.sleep(0.1)
        threading.Thread(target=loop, daemon=True).start()

    def stop(self):
        self._running = False
        self.publisher_.publish(Twist())

class OdomListener(Node):
    def __init__(self):
        super().__init__('odom_listener_gui')
        self.position = (0.0, 0.0)
        self.yaw = 0.0
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.yaw = self.compute_yaw(msg.pose.pose.orientation)

    def compute_yaw(self, q: Quaternion):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

class VelocityListener(Node):
    def __init__(self):
        super().__init__('velocity_listener_gui')
        self.linear = 0.0
        self.angular = 0.0
        self.create_subscription(Twist, '/cmd_vel', self.vel_callback, 10)

    def vel_callback(self, msg):
        self.linear = msg.linear.x
        self.angular = msg.angular.z

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("C-1 Turtlebot3")
        self.setGeometry(100, 100, 1400, 700)
        self.setFocusPolicy(Qt.StrongFocus)

        self.current_angular_display = 0.0
        self.linear_speed = 0.2
        self.angular_speed = 1.0
        self.last_point = None

        self.external_linear = 0.0
        self.external_angular = 0.0
        self.manual_linear = 0.0
        self.manual_angular = 0.0

        self.last_log_time = 0.0 # 로그 출력 시간조절

        rclpy.init(args=None)
        self.cmd_node = CmdVelPublisher()
        self.odom_node = OdomListener()
        self.vel_node = VelocityListener()
        self.toggle_sign_node = SignTogglePublisher() # 표지판 변환 퍼블리시
        self.traffic_sign_node = TrafficSignListener(self.append_log)


        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.traffic_sign_node)
        self.executor.add_node(self.cmd_node)
        self.executor.add_node(self.odom_node)
        self.executor.add_node(self.vel_node)
        self.executor.add_node(self.toggle_sign_node)
        threading.Thread(target=self.executor.spin, daemon=True).start()

        self.label_raw = QLabel("/camera/image_raw/compressed")
        self.label_raw.setFixedSize(640, 360)
        self.label_raw.setAlignment(Qt.AlignCenter)
        self.label_raw.setScaledContents(True)

        self.label_projected = QLabel("/camera/image_projected")
        self.label_projected.setFixedSize(640, 360)
        self.label_projected.setAlignment(Qt.AlignCenter)
        self.label_projected.setScaledContents(True)

        video_row_layout = QHBoxLayout()
        video_row_layout.addWidget(self.label_raw)
        video_row_layout.addWidget(self.label_projected)

        self.label_map = QLabel()
        self.label_map.setAlignment(Qt.AlignCenter)
        self.label_map.setScaledContents(True)
        self.map_path = os.path.join(os.path.dirname(__file__), "map", "course.png")
        self.original_map = QPixmap(self.map_path) if os.path.exists(self.map_path) else QPixmap(520, 496)

        square_size = min(self.original_map.width(), self.original_map.height())
        self.original_map = self.original_map.scaled(QSize(square_size, square_size), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_map.setFixedSize(square_size, square_size)
        self.label_map.setPixmap(self.original_map)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedSize(400, 300)

        self.progress_velocity = QProgressBar()
        self.progress_velocity.setRange(0, 100)
        self.progress_velocity.setValue(0)
        self.progress_velocity.setFormat("속도: %p%")
        self.progress_velocity.setFixedSize(500, 50)

        self.dial_rotation = QDial()
        self.dial_rotation.setRange(-100, 100)
        self.dial_rotation.setValue(0)
        self.dial_rotation.setNotchesVisible(True)
        self.dial_rotation.setEnabled(False)
        self.dial_rotation.setFixedSize(200, 200)

        self.label_rotation_value = QLabel("회전: 0.0")
        self.label_rotation_value.setAlignment(Qt.AlignCenter)
        self.label_rotation_value.setStyleSheet(
    "background: transparent; font-weight: bold; font-size: 16px; margin-left: -500px;"
)

        self.btn_emergency = QPushButton("비상 정지")
        self.btn_emergency.setStyleSheet("background-color: red; color: white;")
        self.btn_emergency.setFixedSize(200, 100)
        self.btn_emergency.setStyleSheet("""
    QPushButton {
        background-color: red;
        color: white;
        border-radius: 5px;
    }
    QPushButton:hover {
        background-color: #cc0000;
    }
    QPushButton:pressed {
        background-color: #990000;
    }
""")
        self.btn_resume = QPushButton("동작 재개")
        self.btn_resume.setStyleSheet("background-color: green; color: white;")
        self.btn_resume.setFixedSize(200, 100)

        self.btn_resume.setStyleSheet("""
    QPushButton {
        background-color: green;
        color: white;
        border-radius: 5px;
    }
    QPushButton:hover {
        background-color: #339933;
    }
    QPushButton:pressed {
        background-color: #006600;
    }
""")

        self.pub_force_stop = self.cmd_node.create_publisher(Bool, '/emergency_stop', 10)
        self.btn_emergency.clicked.connect(lambda: self.pub_force_stop.publish(Bool(data=True)))
        self.btn_resume.clicked.connect(lambda: self.pub_force_stop.publish(Bool(data=False)))

        btn_group = QVBoxLayout()
        btn_group.addWidget(self.btn_emergency)
        btn_group.addWidget(self.btn_resume)

        dial_and_emergency_layout = QHBoxLayout()
        dial_and_emergency_layout.addWidget(self.dial_rotation)
        dial_and_emergency_layout.addLayout(btn_group)
        dial_and_emergency_layout.setAlignment(Qt.AlignCenter)

        center_bottom_layout = QVBoxLayout()
        center_bottom_layout.addWidget(self.progress_velocity)
        center_bottom_layout.addLayout(dial_and_emergency_layout)
        center_bottom_layout.addWidget(self.label_rotation_value)

        self.btn_forward = QPushButton("▲")
        self.btn_backward = QPushButton("▼")
        self.btn_left = QPushButton("◀")
        self.btn_right = QPushButton("▶")
        self.btn_stop = QPushButton("■")
        self.btn_speed_up = QPushButton("가속")
        self.btn_speed_down = QPushButton("감속")

        for btn in [self.btn_forward, self.btn_backward, self.btn_left, self.btn_right, self.btn_stop, self.btn_speed_up, self.btn_speed_down]:
            btn.setFixedSize(90, 90)

        direction_layout = QGridLayout()
        direction_layout.addWidget(self.btn_speed_up, 0, 0)
        direction_layout.addWidget(self.btn_forward, 0, 1)
        direction_layout.addWidget(self.btn_speed_down, 0, 2)
        direction_layout.addWidget(self.btn_left, 1, 0)
        direction_layout.addWidget(self.btn_stop, 1, 1)
        direction_layout.addWidget(self.btn_right, 1, 2)
        direction_layout.addWidget(self.btn_backward, 2, 1)
        direction_layout.setAlignment(Qt.AlignTop)

        direction_widget = QWidget()
        direction_widget.setLayout(direction_layout)
        direction_widget.setFixedHeight(300)

        lower_layout = QHBoxLayout()
        lower_layout.addWidget(self.log_box)

        center_widget = QWidget()
        center_widget.setLayout(center_bottom_layout)
        lower_layout.addWidget(center_widget)
        lower_layout.addWidget(direction_widget)

        left_layout = QVBoxLayout()
        left_layout.addLayout(video_row_layout)
        left_layout.addLayout(lower_layout)
        left_layout.setAlignment(Qt.AlignTop)



        right_layout = QVBoxLayout()
        right_layout.addWidget(self.label_map)
        
        self.btn_toggle_sign = QPushButton("표지판 전환")
        self.btn_toggle_sign.setCheckable(True)
        self.btn_toggle_sign.setFixedSize(150, 40)
        self.btn_toggle_sign.setStyleSheet("background-color: #3366cc; color: white; font-weight: bold;")
        self.btn_toggle_sign.clicked.connect(self.on_toggle_sign)
        right_layout.addWidget(self.btn_toggle_sign, alignment=Qt.AlignRight)

        right_layout.setAlignment(Qt.AlignTop)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        main_layout.setAlignment(Qt.AlignTop)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.ros = ImageROS2Interface()
        self.ros.image_raw_signal.connect(self.update_image_raw)
        self.ros.image_projected_signal.connect(self.update_image_projected)

        self.btn_forward.pressed.connect(lambda: self.manual_drive(self.linear_speed, 0.0))
        self.btn_forward.released.connect(self.stop_manual_drive)
        self.btn_backward.pressed.connect(lambda: self.manual_drive(-self.linear_speed, 0.0))
        self.btn_backward.released.connect(self.stop_manual_drive)
        self.btn_left.pressed.connect(lambda: self.manual_drive(0.0, self.angular_speed))
        self.btn_left.released.connect(self.stop_manual_drive)
        self.btn_right.pressed.connect(lambda: self.manual_drive(0.0, -self.angular_speed))
        self.btn_right.released.connect(self.stop_manual_drive)
        self.btn_stop.clicked.connect(self.stop_manual_drive)
        self.btn_speed_up.clicked.connect(self.increase_speed)
        self.btn_speed_down.clicked.connect(self.decrease_speed)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_map_with_robot)
        self.timer.start(200)

        self.vel_timer = QTimer()
        self.vel_timer.timeout.connect(self.update_velocity_display)
        self.vel_timer.start(200)

        self.setStyleSheet("""
    QMainWindow {
        background-color: black;
    }
    QLabel, QPushButton, QTextEdit, QProgressBar, QDial {
        color: white;
        background-color: #222;
        border: none;
    }
    QPushButton {
        border-radius: 5px;
        padding: 6px;
    }
    QPushButton:hover {
        background-color: #444;
    }
    QPushButton:pressed {
        background-color: #666;
    }
    QProgressBar {
        text-align: center;
    }
""")

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Up:
            self.manual_drive(self.linear_speed, 0.0)
        elif key == Qt.Key_Down:
            self.manual_drive(-self.linear_speed, 0.0)
        elif key == Qt.Key_Left:
            self.manual_drive(0.0, self.angular_speed)
        elif key == Qt.Key_Right:
            self.manual_drive(0.0, -self.angular_speed)

    def keyReleaseEvent(self, event):
        key = event.key()
        if key in [Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right]:
            self.stop_manual_drive()

    def manual_drive(self, linear, angular):
        self.manual_linear = linear
        self.manual_angular = angular
        self.cmd_node.publish_continuous(linear, angular)

    def stop_manual_drive(self):
        self.manual_linear = 0.0
        self.manual_angular = 0.0
        self.cmd_node.stop()

    def update_velocity_display(self):
        self.external_linear = self.vel_node.linear
        self.external_angular = self.vel_node.angular

        linear_total = self.external_linear + self.manual_linear
        angular_total = self.external_angular + self.manual_angular

        alpha = 0.2
        self.current_angular_display = (
            alpha * angular_total + (1 - alpha) * self.current_angular_display
        )

        self.progress_velocity.setValue(int(abs(linear_total) * 100))
        self.dial_rotation.setValue(int(self.current_angular_display * 33))
        self.label_rotation_value.setText(f"회전: {self.current_angular_display:.1f}")

    def update_map_with_robot(self):
        x, y = self.odom_node.position
        yaw = self.odom_node.yaw
        px, py = self.world_to_pixel(x, y)
        pixmap = QPixmap(self.original_map)
        painter = QPainter(pixmap)
        icon_path = os.path.join(os.path.dirname(__file__), "map", "robot_example.png")
        robot_icon = QPixmap(icon_path)
        rotated_icon = robot_icon.transformed(QTransform().rotateRadians(-yaw), Qt.SmoothTransformation)
        icon_w, icon_h = rotated_icon.width(), rotated_icon.height()
        painter.drawPixmap(px - icon_w // 2, py - icon_h // 2, rotated_icon)
        painter.end()
        self.label_map.setPixmap(pixmap)

    def update_image_raw(self, image: QImage):
        self.label_raw.setPixmap(QPixmap.fromImage(image))

    def update_image_projected(self, image: QImage):
        self.label_projected.setPixmap(QPixmap.fromImage(image))

    def world_to_pixel(self, x, y):
        img_w = self.label_map.width()
        img_h = self.label_map.height()
        world_x_min, world_x_max = -2.0, 2.0
        world_y_min, world_y_max = -2.0, 2.0
        px = int((x - world_x_min) / (world_x_max - world_x_min) * img_w)
        py = int((world_y_max - y) / (world_y_max - world_y_min) * img_h) - 16
        return px, py
    
    def on_toggle_sign(self, checked):
        if hasattr(self, "toggle_sign_node"):
            self.toggle_sign_node.toggle(checked)
            self.log_box.append(f"[GUI] 상태 전환됨 → {'왼쪽' if checked else '오른쪽'}")

    def increase_speed(self):
        self.linear_speed = min(self.linear_speed + 0.05, 1.0)
        # self.log_box.append(f"[속도] 증가 → 선속도 {self.linear_speed:.2f}")
        self.manual_drive(math.copysign(self.linear_speed, self.manual_linear), self.manual_angular)

    def decrease_speed(self):
        self.linear_speed = max(self.linear_speed - 0.05, 0.05)
        # self.log_box.append(f"[속도] 감소 → 선속도 {self.linear_speed:.2f}")
        self.manual_drive(math.copysign(self.linear_speed, self.manual_linear), self.manual_angular)

    def append_log(self, text: str):
        self.log_box.append(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
