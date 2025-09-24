import cv2
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
import yaml
import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, Int32
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray


# 호모그래피 좌표변환용 좌표
# 이미지 좌표 (픽셀 단위) - 4개 이상 필요
# img_coords = np.array([
#     [-0.315, 0.021],
#     [-0.430, -0.116],
#     [-0.094, 0.015],
#     [-0.167, -0.106]
# ], dtype=np.float32)

# # 실제 좌표 (m 단위)
# world_coords = np.array([
#     [0.27, 0.11],
#     [0.40, 0.115],
#     [0.27, -0.11],
#     [0.40, -0.115]
# ], dtype=np.float32)

img_coords = np.array([
    [82.000, 225.000],
    [83.000, 172.000],
    [133.000, 224.000],
    [135.000, 172.000],
    [185.000, 224.000],
    [186.000, 172.000],
    [236.000, 224.000],
    [238.000, 172.000],
], dtype=np.float32)
# 실제 좌표 (m 단위)
world_coords = np.array([
    [0.00, 0.00],   # ID-0
    [0.00, 0.06],   # ID-1
    [0.06, 0.00],   # ID-2
    [0.06, 0.06],   # ID-3
    [0.12, 0.00],   # ID-4
    [0.12, 0.06],   # ID-5
    [0.18, 0.00],   # ID-6
    [0.18, 0.06],   # ID-7
], dtype=np.float32)




# aruco 마커 인식 + 위치 추정 + 시각화 데이터를 퍼블리시하는 전체 프로세스를 수행하는 노드
class ArucoMarkerDetector(Node):
    def __init__(self):
        super().__init__('aruco_marker_detector')

        # 압축 이미지 토픽을 구독
        self.subscription = self.create_subscription(
            CompressedImage,  
            'camera/image_raw/compressed',
            # 'image_raw/compressed', 
            self.listener_callback, 
            10)
        self.marker_pub = self.create_publisher(Float32MultiArray, '/marker_pose', 1)

        # 마커의 pose 데이터를 퍼블리시할 퍼블리셔 설정
        self.marker_publisher = self.create_publisher(Marker, 'detected_markers', 10)
        
        self.bridge = CvBridge()
        self.marker_size = 0.04
        self.camera_matrix, self.dist_coeffs = self.load_camera_parameters('calibration_params.yaml')
        
    # aruco 마커를 인식하고 인식 정보를 시각화하는 함수
    def detect_markers(self,image, camera_matrix, dist_coeffs, marker_size):
        # aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000) 
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) # aruco 마커 사전과 파라미터 설정
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        corners, ids, _ = detector.detectMarkers(image) # 이미지에서 마커 감지
        detect_data = []             # 마커 정보 저장 리스트
        coord = [[[0.0, 0.0]]]  # 호모그래피 결과 저장용 초기값

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids) # 감지된 마커 시각화
            rvecs, tvecs, _ = self.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs) # 각 마커에 대해 자세 추정 수행
            
            if rvecs is not None and tvecs is not None:
                for idx, (rvec, tvec, marker_id) in enumerate(zip(rvecs, tvecs, ids)):
                    # 회전벡터를 회전행렬로 변환
                    rot_mat, _ = cv2.Rodrigues(rvec)
                    yaw, pitch, roll = self.rotationMatrixToEulerAngles(rot_mat)

                    # 마커의 위치 정보(x, y, z 값) 추출
                    marker_pos = tvec.flatten()
                    distance = np.linalg.norm(marker_pos)

                    # 마커 데이터 저장
                    detect_data.append([marker_id, marker_pos, (yaw, pitch, roll), distance])

                    # 마커 중심 계산
                    corner = corners[idx][0]
                    center_x = int(np.mean(corner[:, 0]))
                    center_y = int(np.mean(corner[:, 1]))

                    # 이미지에 마커 정보 출력
                    info_text = f"ID:{int(marker_id)}, D:{distance:.2f}m"
                    pos_text = f"X:{marker_pos[0]:.2f}, Y:{marker_pos[1]:.2f}, Z:{marker_pos[2]:.2f}"
                    cv2.putText(image, info_text, (center_x - 40, center_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(image, pos_text, (center_x - 40, center_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    marker_msg = Float32MultiArray()
                    marker_msg.data = [float(marker_id), float(marker_pos[2])]
                    self.marker_pub.publish(marker_msg)
                    # 호모그래피 좌표변환
                    # 이미지 상 좌표와 실제 좌표 간의 변환 행렬 계산
                    H, status = cv2.findHomography(img_coords, world_coords)

                    # 마커 중심 좌표를 변환할 픽셀 좌표로 정의
                    pixel_point = np.array([[marker_pos[0], marker_pos[1]]], dtype=np.float32)
                    pixel_point = np.array([pixel_point])  # shape: (1, 1, 2)

                    # 픽셀 좌표를 실 좌표로 변환
                    coord = cv2.perspectiveTransform(pixel_point, H)

        return image, detect_data, coord


    # aruco 마커의 pose 추정
    def estimatePoseSingleMarkers(self, corners, marker_size, mtx, distortion):
        # 마커의 3D 모델 좌표 설정 (마커의 실제 크기 기준, 중심 기준 좌표)
        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        
        # 회전, 이동 벡터 리스트
        rvecs = []
        tvecs = []

        # 각 마커에 대해 자세 추정
        for c in corners:
            _, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
            rvecs.append(R)
            tvecs.append(t)

        return rvecs, tvecs, []


    # 회전 행렬을 입력으로 받아서 오일러 각도인 r, p, y로 변환해주는 함수
    def rotationMatrixToEulerAngles(self, R):
        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0]) # 회전 행렬의 x, y 평면 요소로부터 계산한 코사인 값 (gimbal lock 여부 판단용)
        
        singular = sy < 1e-6 # 짐벌락 상태 여부 확인

        if not singular: # 일반적인 경우 -> 회전 행렬로부터 Euler 각도 계산
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:            # 짐벌락 상태일 경우 예외처리 -> 일부 값이 0으로 고정
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0

        return np.degrees(x), np.degrees(y), np.degrees(z)


# 카메라 보정 파라미터 파일을 불러와서, 카메라 행렬과 왜곡 계수를 로드하여 반환하는 함수
    def load_camera_parameters(self, yaml_file):
        package_share_directory = get_package_share_directory('aruco_yolo') # aruco_yolo 패키지의 공유 디렉터리 경로를 얻음

        # 보정 파일의 전체 경로 구성
        # calibration_file = os.path.join(package_share_directory, 'config', yaml_file)
        calibration_file = "/home/rokey-jw/rokeypj_ws/src/aruco_yolo/config/calibration_params.yaml"

        # YAML 파일 열어서 데이터 로드
        with open(calibration_file, 'r') as f:
            data = yaml.safe_load(f)
            camera_matrix = np.array(data["camera_matrix"]["data"], dtype=np.float32).reshape(3, 3)
            dist_coeffs = np.array(data["distortion_coefficients"]["data"], dtype=np.float32)
            
        return camera_matrix, dist_coeffs
        
    # 이미지를 subscription하고 aruco 마커를 인식하는 콜백함수
    def listener_callback(self, msg):
        # 압축된 이미지 데이터를 numpy 배열로 변환 후, OpenCV 이미지로 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # aruco 마커 인식 및 위치 추정, 실세계 좌표(coord) 계산
        frame, detect_data, coord = self.detect_markers(frame, self.camera_matrix, self.dist_coeffs, self.marker_size)
        
        # 마커가 하나도 인식되지 않은 경우
        if len(detect_data) == 0:
            self.get_logger().debug("No markers detected")
        else:
            # 가장 가까운 마커 선택 (거리 기준)
            closest_marker = min(detect_data, key=lambda x: x[3])
            self.get_logger().debug(f"Closest Marker ID: {closest_marker[0]}, Distance: {closest_marker[3]:.2f}m")

            # Marker 메시지 배열 생성
            marker_array_msg = MarkerArray()
            for marker in detect_data:
                marker_msg = Marker()
                marker_msg.id = int(marker[0])

                # 실세계 좌표 출력 (호모그래피 변환 결과)
                self.get_logger().info(f"coord.x: {coord[0][0][0]}")
                self.get_logger().info(f"coord.y: {coord[0][0][1]}")

                # 마커의 위치, 방향 정보 설정 (x, y 좌표는 실제 좌표 기준)
                marker_msg.pose.position.x = float(coord[0][0][0])  # marker[1][0]
                marker_msg.pose.position.y = float(coord[0][0][1])  # marker[1][1]
                marker_msg.pose.position.z = marker[1][2]
                marker_msg.pose.orientation.x = marker[2][2]
                marker_msg.pose.orientation.y = marker[2][1]
                marker_msg.pose.orientation.z = marker[2][0]

                # 마커 메시지 배열에 추가
                marker_array_msg.markers.append(marker_msg)

            # 인식된 마커 1개만 퍼블리시
            self.marker_publisher.publish(marker_msg)
            self.get_logger().info(f"========================")

        cv2.imshow('Detected Markers', frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    aruco_marker_detector = ArucoMarkerDetector()
    rclpy.spin(aruco_marker_detector)
    aruco_marker_detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect ArUco markers.')
    parser.add_argument('--marker_size', type=float, default=0.04,
                        help='Size of the ArUco markers in meters.')
    args = parser.parse_args()
    ArucoMarkerDetector.marker_size = args.marker_size
    main()
