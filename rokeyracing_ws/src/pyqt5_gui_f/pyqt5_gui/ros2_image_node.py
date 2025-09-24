import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QImage
import cv2
import numpy as np

class ImageROS2Interface(QObject):
    image_raw_signal = pyqtSignal(QImage)
    image_projected_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()

        # ROS 2 노드 생성 (GUI에서 executor에 등록 필요)
        self.node = Node('dual_image_listener')
        self.bridge = CvBridge()
        self.frame_count = 0

        # compressed 이미지 QoS 적용
        self.sub_raw = self.node.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.cb_image_raw,
            qos_profile_sensor_data
        )

        self.sub_projected = self.node.create_subscription(
            Image,
            '/camera/image_projected',
            self.cb_image_projected,
            10
        )

    def cb_image_raw(self, msg):
        self.frame_count += 1
        if self.frame_count % 3 != 0:
            return
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = cv_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_raw_signal.emit(qt_img)
        except Exception as e:
            print(f"[cb_image_raw] Error: {e}")

    def cb_image_projected(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = cv_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_projected_signal.emit(qt_img)
        except Exception as e:
            print(f"[cb_image_projected] Error: {e}")
