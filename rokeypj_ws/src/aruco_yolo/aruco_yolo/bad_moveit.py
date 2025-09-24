import rclpy
from rclpy.node import Node
from aruco_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from std_msgs.msg import Header
from sensor_msgs.msg import JointState

from geometry_msgs.msg import Twist, Pose, PoseArray
from turtlebot_cosmo_interface.srv import MoveitControl
from aruco_yolo.moveit_client import TurtlebotArmClient
import time
import ast
from std_msgs.msg import Float32MultiArray

class moveit(Node):
    def __init__(self):
        super().__init__('aruco_marker_listener')
       

        self.marker_pub = self.create_subscription(Float32MultiArray, '/marker_pose', self.move_it, 1)
        self.count=0
   
    def move_it(self,msg):
        print("Impossible Mission Start")     
        print("aruco_marker",msg.data)   
        if self.count==0 and msg.data[1]<0.27:
            self.count=1
            arm_client = TurtlebotArmClient()
            # 잡는 위치로 로봇팔 이동 -> 몇 cm로 할것인가? 0.25
            #그리퍼 열어
            time.sleep(2)
            print("Gripper Open")
            response = arm_client.send_request(2, "open")
            arm_client.get_logger().info(f'Response: {response.response}')

            time.sleep(2)
            print("obstacle pick!!")
            response = arm_client.send_request(1, "obstacle_pick") # 좌표 해야함
            arm_client.get_logger().info(f'Response: {response.response}')
            # 그리퍼 오무려
            time.sleep(2)
            print("Gripper Close")
            response = arm_client.send_request(2, "close")
            arm_client.get_logger().info(f'Response: {response.response}')
            time.sleep(2)
            print("obstacle place_route1")
            response = arm_client.send_request(1, "obstacle_place_route1") # 좌표 해야함
            arm_client.get_logger().info(f'Response: {response.response}')
            time.sleep(2)
            print("obstacle place_route2")
            response = arm_client.send_request(1, "obstacle_place_route2") # 좌표 해야함
            arm_client.get_logger().info(f'Response: {response.response}')
            # 놓는 위치로 이동
            time.sleep(2)
            print("obstacle place")
            response = arm_client.send_request(1, "obstacle_place") # 좌표 해야함
            arm_client.get_logger().info(f'Response: {response.response}')
            #그리퍼 열어
            time.sleep(2)
            print("Gripper Open")
            response = arm_client.send_request(2, "open")
            arm_client.get_logger().info(f'Response: {response.response}')
            time.sleep(2)
            print("obstacle place_route2")
            response = arm_client.send_request(1, "obstacle_place_route2") # 좌표 해야함
            arm_client.get_logger().info(f'Response: {response.response}')

            # detect lane 할 수 있는 위치로 이동
            time.sleep(2)
            print("detect lane mode")
            response = arm_client.send_request(1, "lane_tracking_03")
            arm_client.get_logger().info(f'Response: {response.response}')
            self.count=0

def main(args=None):
    rclpy.init(args=args)
    node = moveit()
    # node.move_it()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()