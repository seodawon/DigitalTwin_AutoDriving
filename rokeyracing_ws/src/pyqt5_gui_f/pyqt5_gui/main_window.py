# main_window_final.py
import sys
import math
import threading
import time
from collections import deque
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGridLayout, QWidget
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ros2_image_node import ImageROS2Interface
import os

class GaugeWidget(QWidget):
    def __init__(self, label='속도', unit='m/s', min_value=0.0, max_value=1.0, parent=None):
        super().__init__(parent)
        self.label = label
        self.unit = unit
        self.min_value = min_value
        self.max_value = max_value
        self.current_value = 0.0
        self.setMinimumSize(200, 200)

    def set_value(self, value: float):
        self.current_value = max(self.min_value, min(self.max_value, value))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(10, 10, self.width() - 20, self.height() - 20)
        center = rect.center()
        radius = rect.width() / 2
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(QColor(30, 30, 30))
        painter.drawEllipse(rect)
        painter.setPen(QPen(Qt.white, 2))
        for i in range(11):
            angle = 225 - (270 * i / 10)
            rad = math.radians(angle)
            x1 = center.x() + (radius - 10) * math.cos(rad)
            y1 = center.y() - (radius - 10) * math.sin(rad)
            x2 = center.x() + radius * math.cos(rad)
            y2 = center.y() - radius * math.sin(rad)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
        value_ratio = (self.current_value - self.min_value) / (self.max_value - self.min_value)
        angle = 225 - 270 * value_ratio
        rad = math.radians(angle)
        needle_length = radius - 20
        x = center.x() + needle_length * math.cos(rad)
        y = center.y() - needle_length * math.sin(rad)
        painter.setPen(QPen(Qt.red, 4))
        painter.drawLine(center, QPointF(x, y))
        painter.setPen(Qt.white)
        painter.setFont(QFont('Arial', 10))
        painter.drawText(rect, Qt.AlignTop | Qt.AlignHCenter, self.label)
        painter.drawText(rect, Qt.AlignBottom | Qt.AlignHCenter, f"{self.current_value:.2f} {self.unit}")


class CmdVelPlotter(Node):
    def __init__(self):
        super().__init__('cmd_vel_plotter_gui')
        self.sub_cmd_vel = self.create_subscription(Twist, '/control/cmd_vel', self.cmd_vel_callback, 10)
        self.linear_x_data = deque(maxlen=200)
        self.time_data = deque(maxlen=200)
        self.start_time = time.time()

    def cmd_vel_callback(self, msg):
        t = time.time() - self.start_time
        self.linear_x_data.append(msg.linear.x)
        self.time_data.append(t)


class PIDPlotter(Node):
    def __init__(self):
        super().__init__('pid_plotter_gui')
        self.error_data = deque(maxlen=200)
        self.output_data = deque(maxlen=200)
        self.create_subscription(Float32, '/pid/error', self.error_cb, 10)
        self.create_subscription(Float32, '/pid/output', self.output_cb, 10)

    def error_cb(self, msg): self.error_data.append(msg.data)
    def output_cb(self, msg): self.output_data.append(msg.data)


class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher_gui')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def publish(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub.publish(msg)


class GainPublisher(Node):
    def __init__(self):
        super().__init__('gain_publisher_gui')
        self.kp_pub = self.create_publisher(Float32, '/control/kp', 10)
        self.kd_pub = self.create_publisher(Float32, '/control/kd', 10)

    def publish_kp(self, val): self.kp_pub.publish(Float32(data=val))
    def publish_kd(self, val): self.kd_pub.publish(Float32(data=val))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("C-1 Turtlebot3")
        self.setGeometry(100, 100, 1600, 900)

        rclpy.init()
        self.cmd_node = CmdVelPublisher()
        self.gain_node = GainPublisher()
        self.cmd_plot = CmdVelPlotter()
        self.pid_plot = PIDPlotter()
        self.ros = ImageROS2Interface()

        self.executor = MultiThreadedExecutor()
        for n in [self.cmd_node, self.gain_node, self.cmd_plot, self.pid_plot, self.ros.node]:
            self.executor.add_node(n)
        threading.Thread(target=self.executor.spin, daemon=True).start()

        self.linear_speed = 0.2
        self.angular_speed = 1.0
        self.manual_linear = 0.0
        self.manual_angular = 0.0
        self.kp = 0.0025
        self.kd = 0.0070

        self.label_raw = QLabel("/camera/image_raw/compressed")
        self.label_raw.setStyleSheet("background-color: #111; color: white;")
        self.label_raw.setAlignment(Qt.AlignCenter)

        self.linear_gauge = GaugeWidget("속도", "m/s", 0.0, 1.0)
        self.angular_gauge = GaugeWidget("각속도", "rad/s", -1.82, 1.82)

        self.ros.image_raw_signal.connect(self.update_image_raw)

        self.btn_forward = QPushButton("▲")
        self.btn_backward = QPushButton("▼")
        self.btn_left = QPushButton("◀")
        self.btn_right = QPushButton("▶")
        self.btn_stop = QPushButton("■")
        for btn in [self.btn_forward, self.btn_backward, self.btn_left, self.btn_right, self.btn_stop]:
            btn.setFixedSize(90, 90)

        dir_layout = QGridLayout()
        dir_layout.addWidget(self.btn_forward, 0, 1)
        dir_layout.addWidget(self.btn_left, 1, 0)
        dir_layout.addWidget(self.btn_stop, 1, 1)
        dir_layout.addWidget(self.btn_right, 1, 2)
        dir_layout.addWidget(self.btn_backward, 2, 1)

        self.btn_kp_up = QPushButton("KP ▲")
        self.btn_kp_down = QPushButton("KP ▼")
        self.btn_kd_up = QPushButton("KD ▲")
        self.btn_kd_down = QPushButton("KD ▼")
        self.btn_speed_up = QPushButton("가속 ▲")
        self.btn_speed_down = QPushButton("감속 ▼")
        self.btn_emergency = QPushButton("비상 정지")
        self.btn_resume = QPushButton("동작 재개")

        for b in [self.btn_kp_up, self.btn_kp_down, self.btn_kd_up, self.btn_kd_down,
                  self.btn_speed_up, self.btn_speed_down, self.btn_emergency, self.btn_resume]:
            b.setFixedSize(100, 40)

        self.label_kp_value = QLabel(f"KP: {self.kp:.4f}")
        self.label_kd_value = QLabel(f"KD: {self.kd:.4f}")

        gain_layout = QGridLayout()
        gain_layout.setSpacing(5)
        gain_layout.addWidget(self.btn_kp_up, 0, 0)
        gain_layout.addWidget(self.btn_kd_up, 0, 1)
        gain_layout.addWidget(self.btn_speed_up, 0, 2)
        gain_layout.addWidget(self.btn_kp_down, 1, 0)
        gain_layout.addWidget(self.btn_kd_down, 1, 1)
        gain_layout.addWidget(self.btn_speed_down, 1, 2)
        gain_layout.addWidget(self.label_kp_value, 2, 0)
        gain_layout.addWidget(self.label_kd_value, 2, 1)

        emergency_layout = QHBoxLayout()
        emergency_layout.addWidget(self.btn_emergency)
        emergency_layout.addWidget(self.btn_resume)

        self.cmd_fig = Figure()
        self.cmd_canvas = FigureCanvas(self.cmd_fig)
        self.pid_fig = Figure()
        self.pid_canvas = FigureCanvas(self.pid_fig)

        gauge_layout = QHBoxLayout()
        gauge_layout.addWidget(self.linear_gauge)
        gauge_layout.addWidget(self.angular_gauge)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.label_raw)
        left_layout.addLayout(gauge_layout)
        left_layout.addLayout(dir_layout)

        right_layout = QVBoxLayout()
        right_layout.addLayout(gain_layout)
        right_layout.addLayout(emergency_layout)
        right_layout.addWidget(QLabel("PID 그래프"))
        right_layout.addWidget(self.pid_canvas)
        right_layout.addWidget(QLabel("속도 그래프"))
        right_layout.addWidget(self.cmd_canvas)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.btn_forward.pressed.connect(lambda: self.set_manual(linear=self.linear_speed))
        self.btn_backward.pressed.connect(lambda: self.set_manual(linear=-self.linear_speed))
        self.btn_left.pressed.connect(lambda: self.set_manual(angular=self.angular_speed))
        self.btn_right.pressed.connect(lambda: self.set_manual(angular=-self.angular_speed))
        for b in [self.btn_forward, self.btn_backward, self.btn_left, self.btn_right]:
            b.released.connect(self.reset_manual)
        self.btn_stop.clicked.connect(self.reset_manual)

        self.btn_kp_up.clicked.connect(lambda: self.change_gain('kp', 0.0005))
        self.btn_kp_down.clicked.connect(lambda: self.change_gain('kp', -0.0005))
        self.btn_kd_up.clicked.connect(lambda: self.change_gain('kd', 0.0005))
        self.btn_kd_down.clicked.connect(lambda: self.change_gain('kd', -0.0005))
        self.btn_speed_up.clicked.connect(self.increase_speed)
        self.btn_speed_down.clicked.connect(self.decrease_speed)

        self.btn_emergency.clicked.connect(lambda: self.cmd_node.publish(0.0, 0.0))
        self.btn_resume.clicked.connect(lambda: self.cmd_node.publish(self.manual_linear, self.manual_angular))

        self.graph_timer = QTimer()
        self.graph_timer.timeout.connect(self.update_graphs)
        self.graph_timer.start(200)

        self.vel_timer = QTimer()
        self.vel_timer.timeout.connect(self.update_velocity_display)
        self.vel_timer.start(200)

        self.apply_dark_theme()

    def set_manual(self, linear=None, angular=None):
        if linear is not None: self.manual_linear = linear
        if angular is not None: self.manual_angular = angular
        self.cmd_node.publish(self.manual_linear, self.manual_angular)

    def reset_manual(self): self.set_manual(0.0, 0.0)

    def change_gain(self, param, delta):
        if param == 'kp':
            self.kp = max(0.0, self.kp + delta)
            self.gain_node.publish_kp(self.kp)
            self.label_kp_value.setText(f"KP: {self.kp:.4f}")
        else:
            self.kd = max(0.0, self.kd + delta)
            self.gain_node.publish_kd(self.kd)
            self.label_kd_value.setText(f"KD: {self.kd:.4f}")

    def increase_speed(self): self.set_manual(linear=min(self.manual_linear + 0.05, 1.0))
    def decrease_speed(self): self.set_manual(linear=max(self.manual_linear - 0.05, -1.0))

    def update_velocity_display(self):
        alpha = 0.2
        target_linear = self.manual_linear
        target_angular = self.manual_angular
        if not hasattr(self, 'displayed_linear'): self.displayed_linear = 0.0
        if not hasattr(self, 'displayed_angular'): self.displayed_angular = 0.0
        self.displayed_linear = (1 - alpha) * self.displayed_linear + alpha * target_linear
        self.displayed_angular = (1 - alpha) * self.displayed_angular + alpha * target_angular
        self.linear_gauge.set_value(self.displayed_linear)
        self.angular_gauge.set_value(self.displayed_angular)

    def update_graphs(self):
        self.cmd_fig.clear()
        ax1 = self.cmd_fig.add_subplot(111)
        ax1.plot(self.cmd_plot.time_data, self.cmd_plot.linear_x_data, label='Linear X', color='tab:blue')
        ax1.set_title('cmd_vel.linear.x')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Velocity')
        ax1.legend('Linear X')
        self.cmd_canvas.draw()

        self.pid_fig.clear()
        ax2 = self.pid_fig.add_subplot(111)
        if self.pid_plot.error_data:
            ax2.plot([e / 10.0 for e in self.pid_plot.error_data], label='Error(x0.1)', color='tab:red')
        if self.pid_plot.output_data:
            ax2.plot(self.pid_plot.output_data, label='Output', color='tab:green')
        ax2.set_title('PID Controller')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Value')
        ax2.legend(['Error(x0.1)','Output'])
        self.pid_canvas.draw()

    def update_image_raw(self, image): self.label_raw.setPixmap(QPixmap.fromImage(image))

    def apply_dark_theme(self):
        dark_style = """
        QWidget { background-color: #121212; color: #FFFFFF; }
        QLabel  { color: #FFFFFF; }
        QPushButton {
            background-color: #333333; color: #FFFFFF;
            border: 1px solid #555555; border-radius: 5px; padding: 5px;
        }
        QPushButton:hover { background-color: #444444; }
        QPushButton:pressed { background-color: #555555; }
        """
        self.setStyleSheet(dark_style)


if __name__ == '__main__':
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms/"
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
