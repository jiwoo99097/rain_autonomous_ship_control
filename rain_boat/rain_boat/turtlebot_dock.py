#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist, Quaternion
from global_path_service.srv import GetGlobalPath
from visualization_msgs.msg import Marker
from transforms3d.euler import quat2euler, euler2quat
import math
import csv
import numpy as np
import matplotlib.pyplot as plt

class PurePursuitPathPlanner(Node):
    def __init__(self):
        super().__init__('pure_pursuit_path_planner')
        
        self.start = None
        self.goal = None
        self.global_path = None
        self.current_position = None
        self.current_yaw = 0.0
        self.path_generated = False
        self.look_ahead_distance = 0.3 # Pure Pursuit lookahead distance
        self.linear_velocity = 0.3  # Constant linear velocity
        self.goal_step = 0  # To track if intermediate goal is reached
        self.intermediate_goal = None
        self.docking_path = None

        self.global_path_filename = 'global_path53.csv'
    
        # 노이즈의 표준편차 비율 설정 
        self.linear_noise_ratio = 0.9
        self.angular_noise_ratio = 0.9

        # 오차 데이터 저장을 위한 리스트 초기화
        self.cross_track_errors = [] # 횡단 오차 리스트 
        self.mean_errors = []        # 평균 오차 리스트 
        self.max_errors = []         # 최대 오차 리스트 
        self.rmse_values = []        # RMSE 리스트
       

        # Subscribers
        self.create_subscription(PoseStamped, '/rain/project_autonomous_ship/target_point', self.goal_callback, 10)
        self.create_subscription(PoseStamped, '/current_pose', self.position_callback, 10)

        # Service client to request global path
        self.global_path_service = self.create_service(GetGlobalPath, '/rain/docking_path', self.response_global_path)
        self.path_publisher = self.create_publisher(Path, '/rain/boat_dock/global_path',10)
        self.trajectory_publisher = self.create_publisher(Path, '/rain/boat_dock/trajectory', 10)

        # Publishers
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.marker_publishers = {
            'intermediate_goal': self.create_publisher(Marker, '/rain/boat_dock/intermediate_goal', 10),
            'start_point': self.create_publisher(Marker, '/rain/boat_dock/start_point', 10),
            'goal_point': self.create_publisher(Marker, '/rain/boat_dock/goal_point', 10),
            'current_position': self.create_publisher(PoseStamped, '/rain/boat_dock/current_position', 10),
            'boat_direction': self.create_publisher(Marker, '/rain/boat_dock/boat_direction', 10)
        }
         # Control loop timer
        self.timer = self.create_timer(0.1, self.control_loop)

    def response_global_path(self, request,response):
        self.docking_path = request.path
        self.global_path_callback(self.docking_path)        
        return response     
    
    def global_path_callback(self, path_msg):
        self.global_path = [(pose.pose.position.x, pose.pose.position.y) for pose in path_msg.poses]
        if len(self.global_path) > 1:
            self.intermediate_goal = self.global_path[-21] if len(self.global_path) > 21 else self.global_path[len(self.global_path) // 2]  # Set intermediate goal as midpoint
        self.get_logger().info(f"Received global path with {len(self.global_path)} waypoints")
        self.publish_components()

        self.save_global_path_to_csv(self.global_path_filename)

    def position_callback(self, msg):
        # 시작 지점 설정된 후에만 궤적 생성 
        if self.start is None:
            # 시작 지점이 설정된 순간에 궤적 초기화
            self.start = (msg.pose.position.x, msg.pose.position.y)
            self.get_logger().info(f"Start position set: x: {self.start[0]}, y: {self.start[1]}")
        
            # 궤적 저장 변수 초기화        
            self.trajectory = Path()
            self.trajectory.header.frame_id = "ndt_map"  

        # 현재 위치 업데이트 
        self.current_position = (msg.pose.position.x, msg.pose.position.y)
        orientation_q = msg.pose.orientation
        orientation_list = [orientation_q.w, orientation_q.x, orientation_q.y, orientation_q.z]
        _, _, self.current_yaw = quat2euler(orientation_list)

        self.marker_publishers['current_position'].publish(msg)

        # 궤적 업데이트 
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "ndt_map"
        pose_stamped.pose = msg.pose
        self.trajectory.poses.append(pose_stamped)

        # 궤적 발행 
        self.trajectory.header.stamp = self.get_clock().now().to_msg()
        self.trajectory_publisher.publish(self.trajectory)

    def goal_callback(self, msg):
        # 목표 지점 설정 
        if self.goal is None:
            self.goal = (msg.pose.position.x, msg.pose.position.y)
            self.get_logger().info(f"Goal position received: x: {self.goal[0]}, y: {self.goal[1]}")

    def control_loop(self):
        if self.docking_path :
            self.path_publisher.publish(self.docking_path)

        if self.global_path is None or self.current_position is None:
            return

        # 횡단 오차 계산 및 저장 
        cross_track_error = self.compute_cross_track_error()
        if cross_track_error is not None:
            self.cross_track_errors.append(cross_track_error)

        target_point = self.get_control_target()
        if target_point is None:
            return
        
        self.publish_control_command(target_point)
        self.publish_components()

    def get_control_target(self):
        if self.goal_step == 0 and self.intermediate_goal is not None:
            return self.go_intermediate_goal()
        elif self.goal_step == 1:
            return self.wait_at_intermediate_goal()
        else:
            return self.go_final_goal()

    def go_intermediate_goal(self):
        distance_to_intermediate = self.calculate_distance(self.current_position, self.intermediate_goal)
        if distance_to_intermediate < 0.2:
            self.get_logger().info("Intermediate goal reached. Stopping and waiting for 2 seconds.")
            self.goal_step = 1
            self.stop_moving(self.goal)
            self.stop_time = self.get_clock().now()
            return None
        return self.intermediate_goal
    
    def go_final_goal(self):
        distance_to_goal = self.calculate_distance(self.current_position, self.goal)
        if distance_to_goal < 0.1:
            self.get_logger().info("Final goal reached. Stopping.")
            self.stop_moving(None)
            # 오차 지표 계산 함수 호출
            self.compute_error_metrics()
            return None
        return self.get_target_point()
    
    def wait_at_intermediate_goal(self):
        if (self.get_clock().now() - self.stop_time).nanoseconds / 1e9 < 2.0:
            return None
        else:
            self.get_logger().info("Resuming movement towards final goal.")
            self.goal_step = 2
            return self.get_target_point()
    
    def publish_control_command(self, target_point):
        angular_velocity = self.get_angular_velocity(target_point, self.current_position, self.current_yaw)

        # 노이즈 표준편차 계산 
        linear_std_dev = abs(self.linear_velocity)* self.linear_noise_ratio
        angular_std_dev = abs(angular_velocity)*self.angular_noise_ratio

        # 가우시안 노이즈 추가 
        noisy_linear_velocity = self.linear_velocity + np.random.normal(0, linear_std_dev)
        noisy_angular_velocity = angular_velocity + np.random.normal(0, angular_std_dev)

        cmd_msg = Twist()
        cmd_msg.linear.x = noisy_linear_velocity
        cmd_msg.angular.z = noisy_angular_velocity
        self.publisher_.publish(cmd_msg)

        # cmd_msg = Twist()
        # cmd_msg.linear.x = self.linear_velocity
        # cmd_msg.angular.z = angular_velocity
        # self.publisher_.publish(cmd_msg)

    def publish_components(self):
        self.publish_marker('intermediate_goal', self.intermediate_goal, Marker.SPHERE, [1.0, 0.5, 0.0])
        if self.start is not None:
            self.publish_marker('start_point', self.start, Marker.CUBE, [1.0, 0.0, 0.0])
        if self.goal is not None:
            self.publish_marker('goal_point', self.goal, Marker.CYLINDER, [0.0, 1.0, 0.0])
        if self.current_position is not None:
            self.publish_marker('boat_direction', self.current_position, Marker.ARROW, [0.0, 0.0, 1.0], self.current_yaw)

    def publish_marker(self, marker_type, position, shape, color, yaw=0.0):
        marker = Marker()
        marker.header.frame_id = "ndt_map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = marker_type
        marker.id = 0
        marker.type = shape
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation = self.get_quaternion_from_yaw(yaw)
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = color
        self.marker_publishers[marker_type].publish(marker)
    
    def get_quaternion_from_yaw(self, yaw):
        q = euler2quat(0, 0, yaw)
        return Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])

    def stop_moving(self, target_point=None):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0  

        if target_point is not None:
            angular_velocity = self.get_angular_velocity(target_point, self.current_position, self.current_yaw)
        else:
            angular_velocity = 0.0  

        stop_msg.angular.z = angular_velocity
        self.publisher_.publish(stop_msg)

    def get_target_point(self):
        if not self.global_path:
            return None

        # Find the closest point on the path
        distances = [self.calculate_distance(wp, self.current_position) for wp in self.global_path]
        nearest_index = distances.index(min(distances))

        # Look ahead to find the target point
        for i in range(nearest_index, len(self.global_path)):
            if self.calculate_distance(self.global_path[i], self.current_position) >= self.look_ahead_distance:
                return self.global_path[i]

        return self.global_path[-1]

    def get_angular_velocity(self, target, current_pos, current_yaw):
        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        target_yaw = math.atan2(dy, dx)

        # 현재 방향과 목표점 방향 사이의 각도 차이 
        alpha = target_yaw - current_yaw 
        K = 2 * math.sin(alpha) / self.look_ahead_distance
        return self.linear_velocity * K 

    def calculate_distance(self, point1, point2):
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def compute_cross_track_error(self):
        if self.global_path is None or self.current_position is None:
            return None

        # 현재 위치와 경로의 모든 웨이포인트 사이의 거리 계산
        distances = [self.calculate_distance(self.current_position, wp) for wp in self.global_path]
        cross_track_error = min(distances)
        return cross_track_error

    def compute_error_metrics(self):
        if not self.cross_track_errors:
            return
        
        errors = np.array(self.cross_track_errors)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        rmse = np.sqrt(np.mean(errors**2))

        self.mean_errors.append(mean_error)
        self.max_errors.append(max_error)
        self.rmse_values.append(rmse)

        self.get_logger().info(f"Mean Error: {mean_error}, Max Error: {max_error}, RMSE: {rmse}")
        self.plot_trajectory()

        experiment_number = 5
        trajectory_filename = f"trajectory_experiment_90%_{experiment_number}.csv"
        self.save_trajectory_to_csv(trajectory_filename)

    def save_global_path_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y'])
            for wp in self.global_path:
                x = wp[0]
                y = wp[1]
                writer.writerow([x, y])

    def save_trajectory_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y'])
            for pose in self.trajectory.poses:
                x = pose.pose.position.x
                y = pose.pose.position.y
                writer.writerow([x, y])

    def plot_trajectory(self):
        # 실제 경로 (global_path)
        global_path_x = [wp[0] for wp in self.global_path]
        global_path_y = [wp[1] for wp in self.global_path]

        # 로봇 궤적 (trajectory)
        trajectory_x = [pose.pose.position.x for pose in self.trajectory.poses]
        trajectory_y = [pose.pose.position.y for pose in self.trajectory.poses]

        plt.figure(figsize=(8, 6))
        plt.plot(global_path_x, global_path_y, 'r-', label='Planned Path')
        plt.plot(trajectory_x, trajectory_y, 'b--', label='Actual Trajectory')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Planned Path vs. Actual Trajectory')
        plt.legend()
        plt.grid(True)
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    planner = PurePursuitPathPlanner()
    
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
