#!/usr/bin/env python3
from enum import Enum
import os
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import UInt8
from ament_index_python.packages import get_package_share_directory

class DetectSignCombine(Node):
    def __init__(self):
        super().__init__('detect_signcombine')
        self.prev_detected_sign = None

        self.sub_image_type = 'raw'
        self.pub_image_type = 'compressed'

        if self.sub_image_type == 'compressed':
            self.create_subscription(
                CompressedImage,
                '/detect/image_input/compressed',
                self.cbFindTrafficSign,
                10
            )
        else:
            self.create_subscription(
                Image,
                '/detect/image_input',
                self.cbFindTrafficSign,
                10
            )

        self.pub_traffic_sign = self.create_publisher(UInt8, '/detect/traffic_sign', 10)
        self.pub_image_traffic_sign = self.create_publisher(
            CompressedImage if self.pub_image_type == 'compressed' else Image,
            '/detect/image_output/compressed' if self.pub_image_type == 'compressed' else '/detect/image_output',
            10
        )

        self.cvBridge = CvBridge()
        self.TrafficSign = Enum('TrafficSign', 'red yellow green')
        self.counter = 1

        self.fnPreproc()
        self.get_logger().info('DetectSignCombine Node Initialized')

    def fnPreproc(self):
        self.sift = cv2.SIFT_create()
        package_path = get_package_share_directory('turtlebot3_autorace_detect')
        dir_path = os.path.join(package_path, 'image')

        # 0703 수정!!
        self.templates = {
            'red': {
                'image': cv2.imread(os.path.join(dir_path, 'red.png'), 0),
                'enum': self.TrafficSign.red.value,
                'min_match_cnt': 6,
                'hsv_range': [
                    (np.array([0, 80, 50]),  np.array([10, 255, 255]), 280),
                    (np.array([160, 80, 50]), np.array([179, 255, 255]), 280)
                ]
            },
            'green': {
                'image': cv2.imread(os.path.join(dir_path, 'green.png'), 0),
                'enum': self.TrafficSign.green.value,
                'min_match_cnt': 10,
                'hsv_range': [(np.array([60, 120, 170]), np.array([100, 255, 255]), 320)]
            },

        }

        for key, value in self.templates.items():
            kp, des = self.sift.detectAndCompute(value['image'], None)
            value['kp'] = kp
            value['des'] = des

        FLANN_INDEX_KDTREE = 0
        index_params = {'algorithm': FLANN_INDEX_KDTREE, 'trees': 5}
        search_params = {'checks': 50}
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def fnCalcMSE(self, arr1, arr2):
        return np.sum((arr1 - arr2) ** 2) / (arr1.shape[0] * arr1.shape[1])

    def detect_color_traffic_light(self, image, hsv_ranges):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for lower, upper, thresh in hsv_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            if cv2.countNonZero(mask) > thresh:
                return True
        return False

    def cbFindTrafficSign(self, image_msg):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image_input = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            cv_image_input = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')

        kp1, des1 = self.sift.detectAndCompute(cv_image_input, None)
        if des1 is None or len(kp1) == 0:
            return

        MIN_MSE_DECISION = 60000

        for name, data in self.templates.items():
            if not self.detect_color_traffic_light(cv_image_input, data['hsv_range']):
                self.get_logger().info(f"[Color] {name} 색상 조건 불충족")
                continue

            matches = self.flann.knnMatch(des1, data['des'], k=2)
            good = [m for m, n in matches if m.distance < 0.72 * n.distance]
            self.get_logger().info(f"{name} - Good matches: {len(good)}")

            if len(good) > data['min_match_cnt']:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([data['kp'][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                mse = self.fnCalcMSE(src_pts, dst_pts)

                if mse < MIN_MSE_DECISION:
                    self.get_logger().info(f"############ Detected sign: {name}")
                    if self.prev_detected_sign != data['enum']:
                        msg_sign = UInt8()
                        msg_sign.data = data['enum']
                        self.pub_traffic_sign.publish(msg_sign)
                        self.get_logger().info(f"******** Detected sign: {name}")
                        self.prev_detected_sign = data['enum']
                    else:
                        self.get_logger().info(f"Same sign ({name}) detected. Skipping publish.")

                    final_img = cv2.drawMatches(
                        cv_image_input,
                        kp1,
                        data['image'],
                        data['kp'],
                        good,
                        None,
                        matchColor=(255, 0, 0),
                        singlePointColor=None,
                        matchesMask=None,
                        flags=2
                    )

                    if self.pub_image_type == 'compressed':
                        self.pub_image_traffic_sign.publish(
                            self.cvBridge.cv2_to_compressed_imgmsg(final_img, 'jpg')
                        )
                    else:
                        self.pub_image_traffic_sign.publish(
                            self.cvBridge.cv2_to_imgmsg(final_img, 'bgr8')
                        )
                    break

        # fallback image
        if self.pub_image_type == 'compressed':
            self.pub_image_traffic_sign.publish(
                self.cvBridge.cv2_to_compressed_imgmsg(cv_image_input, 'jpg')
            )
        else:
            self.pub_image_traffic_sign.publish(
                self.cvBridge.cv2_to_imgmsg(cv_image_input, 'bgr8')
            )

def main(args=None):
    rclpy.init(args=args)
    node = DetectSignCombine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# from enum import Enum
# import os
# import cv2
# from cv_bridge import CvBridge
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage, Image
# from std_msgs.msg import UInt8
# from ament_index_python.packages import get_package_share_directory

# class DetectSignCombine(Node):
#     def __init__(self):
#         super().__init__('detect_signcombine')
#         self.prev_detected_sign = None

#         self.sub_image_type = 'raw'
#         self.pub_image_type = 'compressed'

#         if self.sub_image_type == 'compressed':
#             self.create_subscription(
#                 CompressedImage,
#                 '/detect/image_input/compressed',
#                 self.cbFindTrafficSign,
#                 10
#             )
#         else:
#             self.create_subscription(
#                 Image,
#                 '/detect/image_input',
#                 self.cbFindTrafficSign,
#                 10
#             )

#         self.pub_traffic_sign = self.create_publisher(UInt8, '/detect/traffic_sign', 10)
#         self.pub_image_traffic_sign = self.create_publisher(
#             CompressedImage if self.pub_image_type == 'compressed' else Image,
#             '/detect/image_output/compressed' if self.pub_image_type == 'compressed' else '/detect/image_output',
#             10
#         )

#         self.cvBridge = CvBridge()
#         self.TrafficSign = Enum('TrafficSign', 'red yellow green')
#         self.counter = 1

#         self.fnPreproc()
#         self.get_logger().info('DetectSignCombine Node Initialized')

#     def fnPreproc(self):
#         # self.sift = cv2.SIFT_create()
#         # 기존: self.sift = cv2.SIFT_create()
#         self.orb = cv2.ORB_create(nfeatures=1000)

#         package_path = get_package_share_directory('turtlebot3_autorace_detect')
#         dir_path = os.path.join(package_path, 'image')

#         # 0703 수정!!
#         self.templates = {
#             'red': {
#                 'image': cv2.imread(os.path.join(dir_path, 'red.png'), 0),
#                 'enum': self.TrafficSign.red.value,
#                 'min_match_cnt': 12,
#                 'hsv_range': [
#                     (np.array([0, 80, 50]),  np.array([10, 255, 255]), 250),
#                     (np.array([170,80, 50]), np.array([179, 255, 255]), 250)
#                 ]
#             },
#             'green': {
#                 'image': cv2.imread(os.path.join(dir_path, 'green.png'), 0),
#                 'enum': self.TrafficSign.green.value,
#                 'min_match_cnt': 15,
#                 'hsv_range': [(np.array([60, 120, 170]), np.array([100, 255, 255]), 300)]
#             },

#         }

#         for key, value in self.templates.items():
#             # kp, des = self.sift.detectAndCompute(value['image'], None)
#             kp, des = self.orb.detectAndCompute(value['image'], None)
#             value['kp'] = kp
#             value['des'] = des

#         # FLANN_INDEX_KDTREE = 0
#         # index_params = {'algorithm': FLANN_INDEX_KDTREE, 'trees': 5}
#         # search_params = {'checks': 50}
#         # self.flann = cv2.FlannBasedMatcher(index_params, search_params)
#         self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

#     def fnCalcMSE(self, arr1, arr2):
#         return np.sum((arr1 - arr2) ** 2) / (arr1.shape[0] * arr1.shape[1])

#     def detect_color_traffic_light(self, image, hsv_ranges):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         for lower, upper, thresh in hsv_ranges:
#             mask = cv2.inRange(hsv, lower, upper)
#             if cv2.countNonZero(mask) > thresh:
#                 return True
#         return False

#     def cbFindTrafficSign(self, image_msg):
#         if self.counter % 3 != 0:
#             self.counter += 1
#             return
#         else:
#             self.counter = 1

#         if self.sub_image_type == 'compressed':
#             np_arr = np.frombuffer(image_msg.data, np.uint8)
#             cv_image_input = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#         else:
#             cv_image_input = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')

#         # kp1, des1 = self.sift.detectAndCompute(cv_image_input, None)
#         kp1, des1 = self.orb.detectAndCompute(cv_image_input, None)
#         # if des1 is None or len(kp1) == 0:
#         #     return
#         if data['des'] is None or len(data['des']) == 0:
#             return

#         # MIN_MSE_DECISION = 55000
#         MIN_MSE_DECISION = 50000

#         for name, data in self.templates.items():
#             if not self.detect_color_traffic_light(cv_image_input, data['hsv_range']):
#                 self.get_logger().info(f"[Color] {name} 색상 조건 불충족")
#                 continue

#             # matches = self.flann.knnMatch(des1, data['des'], k=2)
#             matches = self.matcher.knnMatch(des1, data['des'], k=2)
#             good = [m for m, n in matches if m.distance < 0.7 * n.distance]
#             self.get_logger().info(f"{name} - Good matches: {len(good)}")

#             if len(good) > data['min_match_cnt']:
#                 src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#                 dst_pts = np.float32([data['kp'][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#                 mse = self.fnCalcMSE(src_pts, dst_pts)

#                 # if mse < MIN_MSE_DECISION:
#                 self.get_logger().info(f"############ Detected sign: {name}")
#                 if self.prev_detected_sign != data['enum']:
#                     msg_sign = UInt8()
#                     msg_sign.data = data['enum']
#                     self.pub_traffic_sign.publish(msg_sign)
#                     self.get_logger().info(f"******** Detected sign: {name}")
#                     self.prev_detected_sign = data['enum']
#                 else:
#                     self.get_logger().info(f"Same sign ({name}) detected. Skipping publish.")

#                 final_img = cv2.drawMatches(
#                     cv_image_input,
#                     kp1,
#                     data['image'],
#                     data['kp'],
#                     good,
#                     None,
#                     matchColor=(255, 0, 0),
#                     singlePointColor=None,
#                     matchesMask=None,
#                     flags=2
#                 )

#                 if self.pub_image_type == 'compressed':
#                     self.pub_image_traffic_sign.publish(
#                         self.cvBridge.cv2_to_compressed_imgmsg(final_img, 'jpg')
#                     )
#                 else:
#                     self.pub_image_traffic_sign.publish(
#                         self.cvBridge.cv2_to_imgmsg(final_img, 'bgr8')
#                     )
#                 break

#         # fallback image
#         if self.pub_image_type == 'compressed':
#             self.pub_image_traffic_sign.publish(
#                 self.cvBridge.cv2_to_compressed_imgmsg(cv_image_input, 'jpg')
#             )
#         else:
#             self.pub_image_traffic_sign.publish(
#                 self.cvBridge.cv2_to_imgmsg(cv_image_input, 'bgr8')
#             )

# def main(args=None):
#     rclpy.init(args=args)
#     node = DetectSignCombine()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()