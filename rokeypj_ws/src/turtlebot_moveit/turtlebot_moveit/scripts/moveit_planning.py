#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from moveit_msgs.action import MoveGroup
from builtin_interfaces.msg import Duration
import math
from moveit_msgs.msg import MoveItErrorCodes

class IKMoveExample(Node):
    def __init__(self):
        super().__init__('ik_move_example')

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            '/move_action'  # MoveGroup action 이름 (보통 기본값)
        )

    def send_goal_to_movegroup(self, target_pose: PoseStamped):
        self.get_logger().info('MoveGroup 액션 goal 생성 중...')

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = 'arm'  # 사용 중인 MoveIt 그룹 이름

        req.allowed_planning_time = 10.0
        req.num_planning_attempts = 10
        req.max_velocity_scaling_factor = 0.5
        req.max_acceleration_scaling_factor = 0.5

        # PositionConstraint로 목표 pose 설정 (간단히 BoundingVolume으로 타겟 지정)
        pos_constraint = PositionConstraint()
        pos_constraint.header = target_pose.header
        pos_constraint.link_name = 'joint4'  # end-effector link 이름
        pos_constraint.target_point_offset.x = 0.0
        pos_constraint.target_point_offset.y = 0.0
        pos_constraint.target_point_offset.z = 0.0

        # BoundingVolume으로 정확한 점 지정 (AABB 형태지만 min/max 같은 값으로 점 지정 가능)
        pos_constraint.constraint_region = BoundingVolume()
        pos_constraint.constraint_region.primitives.append(
            self.create_box_primitive(0.001, 0.001, 0.001)
        )
        pos_constraint.constraint_region.primitive_poses.append(target_pose.pose)
        
        req.goal_constraints.append(Constraints(position_constraints=[pos_constraint]))

        # Orientation 제약 조건 추가 가능
        # ori_constraint = OrientationConstraint()
        # req.goal_constraints[-1].orientation_constraints.append(ori_constraint)

        req.allowed_planning_time = 5.0
        goal_msg.request = req

        self.get_logger().info('[플래닝] 목표 pose로 MoveGroup goal 전송 준비')

        if not self.move_group_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup 액션 서버를 찾을 수 없습니다.')
            return

        self.get_logger().info('MoveGroup 액션 서버 연결됨, goal 전송')
        send_goal_future = self.move_group_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def create_box_primitive(self, x, y, z):
        from shape_msgs.msg import SolidPrimitive
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [x, y, z]
        return box

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('MoveGroup goal rejected!')
            return

        self.get_logger().info('MoveGroup goal accepted. 결과 대기 중...')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if not result or not result.planned_trajectory or not result.planned_trajectory.joint_trajectory.points:
            error_code = result.error_code.val
            self.get_logger().info(f"플래닝 결과 코드: {error_code}")
            if error_code != MoveItErrorCodes.SUCCESS:
                self.get_logger().error('플래닝 실패!')
            self.get_logger().error('플래닝 실패!')
            return
        self.get_logger().info('플래닝 성공! Trajectory point 개수: {}'.format(len(result.planned_trajectory.joint_trajectory.points)))
        # 필요하면 여기서 trajectory 실행 노드로 전달 가능

def main(args=None):
    rclpy.init(args=args)
    node = IKMoveExample()

    # 이동할 목표 pose 정의
    target_pose = PoseStamped()
    target_pose.header.frame_id = 'base_link'
    target_pose.pose.position.x = 0.2  # meter 단위 #  이거 좌표 낼 설정해봐야함
    target_pose.pose.position.y = 0.0
    target_pose.pose.position.z = 0.15
    # 회전은 간단히 단위 quaternion
    target_pose.pose.orientation.w = 1.0

    node.send_goal_to_movegroup(target_pose)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
