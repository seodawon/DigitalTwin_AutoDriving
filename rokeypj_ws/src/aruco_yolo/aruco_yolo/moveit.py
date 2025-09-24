
import rclpy
from rclpy.node import Node
from turtlebot_cosmo_interface.srv import MoveitControl
from geometry_msgs.msg import Pose, PoseArray
import time
from std_msgs.msg import Float32MultiArray


class TurtlebotArmClient(Node):

    def __init__(self):
        super().__init__('turtlebot_arm_client')
        self.client = self.create_client(MoveitControl, 'moveit_control')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = MoveitControl.Request()
        self.marker_pub = self.create_subscription(Float32MultiArray, '/marker_pose', self.move_it, 10)
        self.count=0
    def send_request(self, cmd, posename='', waypoints=None):
        self.req.cmd = cmd
        self.req.posename = posename
        if waypoints:
            self.req.waypoints = waypoints
        self.future = self.client.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
    
    def move_it(self,msg):
        print("Impossible Mission Start")        
        # 잡는 위치로 로봇팔 이동 -> 몇 cm로 할것인가?
        print("aruco_marker",msg.data)   
        if self.count==0:
            self.count=1
            time.sleep(2)
            print("obstacle pick!!")
            response = self.send_request(2, "obstacle_pick") # 좌표 해야함
            self.get_logger().info(f'Response: {response.response}')
            # 그리퍼 오무려
            time.sleep(2)
            print("Gripper Close")
            response = self.send_request(2, "close")
            self.get_logger().info(f'Response: {response.response}')
            # 놓는 위치로 이동
            time.sleep(2)
            print("obstacle place")
            response = self.send_request(2, "obstacle_place") # 좌표 해야함
            self.get_logger().info(f'Response: {response.response}')
            #그리퍼 열어
            time.sleep(2)
            print("Gripper Open")
            response = self.send_request(2, "open")
            self.get_logger().info(f'Response: {response.response}')


            # detect lane 할 수 있는 위치로 이동
            time.sleep(2)
            print("detect lane mode")
            response = self.send_request(2, "lane_tracking_03")
            self.get_logger().info(f'Response: {response.response}')

def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotArmClient()
    # node.move_it()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()