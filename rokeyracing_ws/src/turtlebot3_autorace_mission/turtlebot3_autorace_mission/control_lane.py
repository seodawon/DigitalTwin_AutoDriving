#!/usr/bin/env python3
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
# Author: Leon Jung, Gilbert, Ashe Kim, Hyungyu Kim, ChanHyeong Lee

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import Float64, Float32
from nav_msgs.msg import Odometry
import math
import time
from std_msgs.msg import UInt8
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from aruco_yolo.moveit_client import TurtlebotArmClient # error?
from std_msgs.msg import Float32MultiArray

class ControlLane(Node):

    def __init__(self):
        super().__init__('control_lane')
        self.group = ReentrantCallbackGroup()
        self.sub_lane = self.create_subscription(
            Float64,
            '/control/lane',
            self.callback_follow_lane,
            1,callback_group=self.group
        )
        
        self.sub_max_vel = self.create_subscription(
            Float64,
            '/control/max_vel',
            self.callback_get_max_vel,
            1
        )
                
        self.pub_cmd_vel = self.create_publisher(
            Twist,
            '/control/cmd_vel',
            1,callback_group=self.group
        )
        
        self.sub_sign = self.create_subscription(
            UInt8,
            '/detect/traffic_sign',
            self.sign_callback,
            10,callback_group=self.group
        )
        
        # 비상 정지 토픽 구독
        self.sub_force_stop = self.create_subscription( 
            Bool,
            '/emergency_stop',
            self.callback_force_stop,
            1
        )
        #dawon
        self.marker_pub = self.create_subscription(Float32MultiArray, '/marker_pose', self.move_it, 1)
        self.count=0

        self.error_pub = self.create_publisher(Float32, 'pid/error', 10)
        self.output_pub = self.create_publisher(Float32, 'pid/output', 10)

        self.marker_pub = self.create_subscription(Float32MultiArray, '/marker_pose', self.move_it, 10)
        self.count=0
        # PD control related variables
        self.last_error = 0
        self.MAX_VEL = 0.3
        
        self.start_count = True
        self.start =True
        self.force_stop = False # 비상 정지를 위한 flag 변수
    def move_it(self,msg):
        print("Impossible Mission Start")     
        print("aruco_marker",msg.data)   
        if self.count==0 and msg.data[1] < 0.27: #0.27
            self.get_logger().info(f'marker_distance_value: {msg.data[1]}!!!!')
            self.start = False # 정지해!
            self.shut_down() 
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
            # self.count=0
            self.start = True # start해!
            time.sleep(2)

   
    def callback_get_max_vel(self, max_vel_msg):
        self.MAX_VEL = max_vel_msg.data
        
    # red: 1, yellow: 2, green: 3, close:9, open:10
    def sign_callback(self,msg):
        self.get_logger().info(f"{msg}")
        if msg.data == 1: # 빨간불
            self.start = False # 정지해!
        
        elif msg.data == 3: # 초록불
            self.start = True # 시작해!
            
        elif msg.data == 9: # 차단바 close
            self.start = False # stop 명령
        
        elif msg.data == 10: # 차단바 open
            self.start = True # 다시 시작!
                        
    def euler_from_quaternion(self, msg):
        """
        Convert a quaternion into Euler angles (roll, pitch, yaw).

        msg: geometry_msgs.msg.Quaternion.
        return: (roll, pitch, yaw) tuple.
        """
        x = msg[0]
        y = msg[1]
        z = msg[2]
        w = msg[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = max(min(t2, 1.0), -1.0)
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw
    
    def callback_follow_lane(self, desired_center):
        """
        Receive lane center data to generate lane following control commands.
        """
        
        if self.force_stop or not self.start:
            self.shut_down()    # 비상 정지 시 주행 X
            return

        center = desired_center.data
        error = center - 500

        Kp = 0.005
        Kd = 0.006

        angular_z = Kp * error + Kd * (error - self.last_error)
        self.last_error = error

        self.error_pub.publish(Float32(data=error))
        self.output_pub.publish(Float32(data=angular_z))

        twist = Twist()
        # Linear velocity: adjust speed based on error (maximum 0.05 limit)
        twist.linear.x = min(self.MAX_VEL * (max(1 - abs(error) / 500, 0) ** 2.2), 0.4) # 0.05
        twist.angular.z = -max(angular_z, -0.8) if angular_z < 0 else -min(angular_z, 0.8) # 1.0 -> 2.0으로 늘릴 경우 회전에 대해서 빠르기는 하지만 차선 인지를 잘 못한다.
        self.pub_cmd_vel.publish(twist)

    # 비상 정지 콜백 함수
    def callback_force_stop(self, msg):
        self.force_stop = msg.data
        if self.force_stop:
            self.get_logger().warn("비상 정지 활성화됨")
            self.shut_down()
        else:
            self.get_logger().info("비상 정지 해제됨")
            
    def shut_down(self):
        self.get_logger().info('Shutting down. cmd_vel will be 0')
        twist = Twist()
        self.pub_cmd_vel.publish(twist)


# def main(args=None):
#     rclpy.init(args=args)
#     node = ControlLane()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.shut_down()
#         node.destroy_node()
#         rclpy.shutdown()

# 멀티 스레드용 main 함수
def main(args=None):
    rclpy.init(args=args)
    node = ControlLane()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        # rclpy.spin(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.shut_down()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()