#!/usr/bin/env python
#
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors:
#   - Leon Jung, Gilbert, Ashe Kim, ChanHyeong Lee
#   - [AuTURBO] Kihoon Kim (https://github.com/auturbo)

from enum import Enum
import math
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_msgs.msg import UInt8

from collections import deque

CLOSE_STATE = 9
OPEN_STATE = 10

def fnCalcDistanceDot2Line(a, b, c, x0, y0):
    distance = abs(x0 * a + y0 * b + c) / math.sqrt(a * a + b * b)
    return distance


def fnCalcDistanceDot2Dot(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def fnArrangeIndexOfPoint(arr):
    new_arr = arr[:]
    arr_idx = list(range(len(arr)))
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if new_arr[i] < new_arr[j]:
                new_arr[i], new_arr[j] = new_arr[j], new_arr[i]
                arr_idx[i], arr_idx[j] = arr_idx[j], arr_idx[i]
    return arr_idx


def fnCheckLinearity(point1, point2, point3):
    threshold_linearity = 50
    x1, y1 = point1
    x2, y2 = point3
    if x2 - x1 != 0:
        a = (y2 - y1) / (x2 - x1)
    else:
        a = 1000
    b = -1
    c = y1 - a * x1
    err = fnCalcDistanceDot2Line(a, b, c, point2[0], point2[1])
    return err < threshold_linearity


def fnCheckDistanceIsEqual(point1, point2, point3):
    threshold_distance_equality = 3
    distance1 = fnCalcDistanceDot2Dot(point1[0], point1[1], point2[0], point2[1])
    distance2 = fnCalcDistanceDot2Dot(point2[0], point2[1], point3[0], point3[1])
    std = np.std([distance1, distance2])
    return std < threshold_distance_equality


# ------------------------
# ROS2 Node: DetectLevelNode
# ------------------------
class DetectLevelNode(Node):

    def __init__(self):
        super().__init__('detect_level')
        self.get_logger().info('Starting detect_level node (ROS2)')
        
        self.level_state_buffer = deque(maxlen=5)
        self.last_state = None  # 'open' or 'close'
        self.has_seen_horizontal_line = False # 0702 수정 !!!
        
        hue_range = IntegerRange(from_value=0, to_value=179, step=1)
        sat_range = IntegerRange(from_value=0, to_value=255, step=1)
        light_range = IntegerRange(from_value=0, to_value=255, step=1)

        hue_l_descriptor = ParameterDescriptor(
            description='Lower hue threshold',
            integer_range=[hue_range]
        )
        hue_h_descriptor = ParameterDescriptor(
            description='Upper hue threshold',
            integer_range=[hue_range]
        )
        sat_l_descriptor = ParameterDescriptor(
            description='Lower saturation threshold',
            integer_range=[sat_range]
        )
        sat_h_descriptor = ParameterDescriptor(
            description='Upper saturation threshold',
            integer_range=[sat_range]
        )
        light_l_descriptor = ParameterDescriptor(
            description='Lower value (lightness) threshold',
            integer_range=[light_range]
        )
        light_h_descriptor = ParameterDescriptor(
            description='Upper value (lightness) threshold',
            integer_range=[light_range]
        )

        # delcare parameters
        self.declare_parameter('detect.level.red.hue_l', 0, descriptor=hue_l_descriptor)
        self.declare_parameter('detect.level.red.hue_h', 24, descriptor=hue_h_descriptor)
        self.declare_parameter('detect.level.red.saturation_l', 11, descriptor=sat_l_descriptor)
        self.declare_parameter('detect.level.red.saturation_h', 255, descriptor=sat_h_descriptor)
        self.declare_parameter('detect.level.red.lightness_l', 50, descriptor=light_l_descriptor)
        self.declare_parameter('detect.level.red.lightness_h', 255, descriptor=light_h_descriptor)

        self.declare_parameter('is_detection_calibration_mode', False)

        # get parameters
        self.hue_red_l = self.get_parameter('detect.level.red.hue_l').value
        self.hue_red_h = self.get_parameter('detect.level.red.hue_h').value
        self.saturation_red_l = self.get_parameter('detect.level.red.saturation_l').value
        self.saturation_red_h = self.get_parameter('detect.level.red.saturation_h').value
        self.lightness_red_l = self.get_parameter('detect.level.red.lightness_l').value
        self.lightness_red_h = self.get_parameter('detect.level.red.lightness_h').value
        self.is_calibration_mode = self.get_parameter('is_detection_calibration_mode').value

        self.add_on_set_parameters_callback(self.on_parameter_change)

        self.sub_image_type = 'raw'  # 'raw' or 'compressed'
        self.pub_image_type = 'compressed'  # 'raw' or 'compressed'

        self.StepOfLevelCrossing = Enum('StepOfLevelCrossing', 'pass_level exit')

        self.is_level_crossing_finished = False
        self.stop_bar_count = 0
        self.counter = 1
        self.cv_image = None

        self.cv_bridge = CvBridge()

        # create publishers
        if self.pub_image_type == 'compressed':
            self.pub_image_level = self.create_publisher(
                CompressedImage, '/detect/image_output/compressed', 10)
            if self.is_calibration_mode:
                self.pub_image_color_filtered = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub1/compressed', 10)
        else:  # raw
            self.pub_image_level = self.create_publisher(
                Image, '/detect/image_output', 10)
            if self.is_calibration_mode:
                self.pub_image_color_filtered = self.create_publisher(
                    Image, '/detect/image_output_sub1', 10)

        # create subscribers
        if self.sub_image_type == 'compressed':
            self.create_subscription(
                CompressedImage, '/detect/image_input/compressed', self.get_image, 10)
        else:  # raw
            self.create_subscription(
                Image, '/detect/image_input', self.get_image, 10)

        # self.pub_level_opened = self.create_publisher(UInt8, '/level_opened', 10)
        self.pub_level_opened = self.create_publisher(UInt8, '/detect/traffic_sign', 10)
        
        self.timer = self.create_timer(1.0/15.0, self.timer_callback)

        time.sleep(1.0)

    def on_parameter_change(self, params):
        for param in params:
            if param.name == 'detect.level.red.hue_l':
                self.hue_red_l = param.value
            elif param.name == 'detect.level.red.hue_h':
                self.hue_red_h = param.value
            elif param.name == 'detect.level.red.saturation_l':
                self.saturation_red_l = param.value
            elif param.name == 'detect.level.red.saturation_h':
                self.saturation_red_h = param.value
            elif param.name == 'detect.level.red.lightness_l':
                self.lightness_red_l = param.value
            elif param.name == 'detect.level.red.lightness_h':
                self.lightness_red_h = param.value
        self.get_logger().info('Dynamic parameters updated.')
        return SetParametersResult(successful=True)

    def timer_callback(self):
        if self.cv_image is not None:
            self.find_level()

    def get_image(self, image_msg):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            try:
                self.cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error('CV Bridge error: %s' % str(e))

    def mask_red_of_level(self):
        image = np.copy(self.cv_image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # lower_red = np.array([self.hue_red_l, self.saturation_red_l, self.lightness_red_l])
        # upper_red = np.array([self.hue_red_h, self.saturation_red_h, self.lightness_red_h])
        
        # 빨간색은 두 개의 Hue 범위로 나뉠 수 있음:
        lower_red1 = np.array([0, 100, 70])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([160, 100, 70])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)


        # mask = cv2.inRange(hsv, lower_red, upper_red)
        
        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_color_filtered.publish(
                    self.cv_bridge.cv2_to_compressed_imgmsg(mask, 'jpg'))
            else:
                self.pub_image_color_filtered.publish(
                    self.cv_bridge.cv2_to_imgmsg(mask, 'mono8'))
                
        mask = cv2.bitwise_not(mask)
        # OpenCV로 마스크 시각화
        cv2.imshow("Red Mask", mask)
        cv2.waitKey(1)
        
        return mask
    
    # 0702 수정 !!! (차단바가 열리면 보이지 않는 문제 -> 열림/닫힘을 수평선 기준으로 판단)
    def find_level(self):
        mask = self.mask_red_of_level()
        if cv2.countNonZero(mask) == 0:
            self.get_logger().info('@@@ Mask is empty.')
            return

        frame = self.cv_image.copy()
        h, w = mask.shape[:2]

        # ROI: 오른쪽 아래 직사각형
        roi_x1 = w // 3
        roi_x2 = w
        roi_y1 = 0# h // 3
        roi_y2 = h

        roi = mask[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_for_draw = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # 선분 추출
        edges = cv2.Canny(roi, 70, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=30, minLineLength=30, maxLineGap=15)

        horizontal_lines = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                slope = dy / (dx + 1e-6)
                length = math.hypot(dx, dy)

                # 시각화
                cv2.line(roi_for_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # 수평선 판단
                if abs(slope) < 0.5:
                    horizontal_lines += 1

        # 수평선이 2개 이상 감지되면 닫힘 상태로 기록
        if horizontal_lines >= 2:
            self.has_seen_horizontal_line = True

        # 수평선이 없거나 선이 아예 없을 경우 → 열린 상태로 판정
        if lines is None or horizontal_lines <= 1:
            if self.has_seen_horizontal_line:
                is_open = True
                self.get_logger().info("[ROI] 차단바 수평선 사라짐 → open 판단")
            else:
                is_open = False
        else:
            is_open = False

        # 상태 버퍼에 기록
        self.level_state_buffer.append('open' if is_open else 'close')

        # 열린 상태가 충분히 누적되면 has_seen 초기화
        # -> 5 번 판단 이후에 close 판단으로 돌아갈 가능성이 있어서 못쓸 것 같음
        # if self.level_state_buffer.count('open') >= 5:
        #     self.has_seen_horizontal_line = False

        open_count = self.level_state_buffer.count('open')
        close_count = self.level_state_buffer.count('close')

        # 상태 변화 조건
        if self.last_state == 'open':
            if close_count >= 3:
                self.last_state = 'close'
                self.pub_level_opened.publish(UInt8(data=CLOSE_STATE))
                self.get_logger().info("[차단바] 안정적으로 close 판정")
        elif self.last_state == 'close':
            if open_count >= 4:
                self.last_state = 'open'
                self.pub_level_opened.publish(UInt8(data=OPEN_STATE))
                self.get_logger().info("[차단바] 안정적으로 open 판정")
        else:
            if open_count >= 4:
                self.last_state = 'open'
                self.pub_level_opened.publish(UInt8(data=OPEN_STATE))
            elif close_count >= 3:
                self.last_state = 'close'
                self.pub_level_opened.publish(UInt8(data=CLOSE_STATE))

        # 디버그 시각화
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        cv2.putText(roi_for_draw, "ROI", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if self.pub_image_type == 'compressed':
            comp_img_msg = self.cv_bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpg')
            self.pub_image_level.publish(comp_img_msg)
        else:
            img_msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.pub_image_level.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DetectLevelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
