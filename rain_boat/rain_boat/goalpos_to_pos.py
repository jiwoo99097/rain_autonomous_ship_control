#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from goal_position_msg.msg import GoalPosition  # 커스텀 메시지 가져오기
from transforms3d.euler import euler2quat

class GoalPostoPoseStamped(Node):
    def __init__(self):
        super().__init__('pose_stamped_to_goal_position')

        # '/rain/project_autonomous_ship/target_point' 토픽 구독
        self.subscription = self.create_subscription(
            GoalPosition,
            '/rain/goal_position',
            self.goal_position_callback,
            10
        )
        
        self.goal_position_publisher = self.create_publisher(PoseStamped, '/rain/project_autonomous_ship/target_point', 10)

    def goal_position_callback(self, msg):
        # GoalPosition 메시지에서 x, y, yaw 값 추출
        x = msg.x
        y = msg.y
        yaw = msg.yaw

        # PoseStamped 메시지 생성
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "ndt_map"  # 필요한 프레임 ID로 설정

        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = 0.0

        # Yaw를 쿼터니언으로 변환
        quaternion = euler2quat(0, 0, yaw)
        pose_stamped.pose.orientation.x = quaternion[1]
        pose_stamped.pose.orientation.y = quaternion[2]
        pose_stamped.pose.orientation.z = quaternion[3]
        pose_stamped.pose.orientation.w = quaternion[0]

        # 변환된 PoseStamped 메시지 퍼블리시
        self.goal_position_publisher.publish(pose_stamped)
        # self.get_logger().info(f"Published PoseStamped: x={x}, y={y}, yaw={yaw}")

def main(args=None):
    rclpy.init(args=args)
    node = GoalPostoPoseStamped()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
