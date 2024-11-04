#!/usr/bin/env python3

# ros2 launch vrx_gz competition.launch.py world:=sydney_regatta

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Path
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import Point, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster
from pyproj import Transformer
from global_path_service.srv import GetGlobalPath
from tf_transformations import euler_from_quaternion
import math 
import numpy as np 
from functools import partial
import time


class DockingPathFollower(Node):
    def __init__(self):
        super().__init__('docking_path_follower')
        self._init_parameters()
        self._init_publishers_subscribers()
        self.timer = self.create_timer(1.0 / self.control_frequency, self.control_loop)
        self.position_history = [] # 위치 이력 저장 리스트 
        self.global_path = None
        self.current_waypoint_index = 0
        self.obstacles = {} # 딕셔너리 사용 
        self.goal_reached = False
        self.start_time = None

        self.intermediate_goal_reached = False  # 중간 목표에 도달했는지 확인하는 플래그
        self.intermediate_goal = None  # 중간 목표를 저장
        self.intermediate_goal_threshold = 1.0  # 중간 목표 도달 기준 (거리)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.global_path_client = self.create_client(GetGlobalPath, '/rain/docking_path')
        while not self.global_path_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('get_global_path service not available, waiting again...')
        self.request_global_path()
         
    def declare_and_get_parameters(self):
        params = [
            ('goal_lat', -33.72219231290662),
            ('goal_lon', 150.67352145568518),
            ('intermediate_goal_lat', -33.722213717246916),
            ('intermediate_goal_lon', 150.67385537274836),
            ('linear_velocity', 0.5),
            ('initial_k_value', 1.0),
            ('goal_threshold', 1.0),
            ('av_coefficient', -1000),
            ('max_thrust', 300),
            ('look_ahead_distance', 7.0),
            ('control_frequency', 10.0),
            ('v_noise_std', 0.1),  # 선형 속도 노이즈의 표준편차
            ('w_noise_std', 0.1),   # 각속도 노이즈의 표준편차
        ]
        for name, default in params:
            self.declare_parameter(name, default)
            setattr(self, name, self.get_parameter(name).value)

    def _init_parameters(self):
        self.declare_and_get_parameters()
        self.utm_transformer = Transformer.from_crs("EPSG:4326", "EPSG:7856", always_xy=True)
        self.goal = self.wgs84_to_utm_sydney(self.goal_lon, self.goal_lat)
        self.intermediate_goal = self.wgs84_to_utm_sydney(self.intermediate_goal_lon, self.intermediate_goal_lat)
        self.current_pos = (0.0, 0.0)
        self.previous_pos = (0.0, 0.0)
        self.current_velocity = 0.0
        self.current_yaw = 0.0
        self.left_thrust = 0.0
        self.right_thrust = 0.0
        self.k_value = self.initial_k_value
        self.waypoints = None
        self.start = None

    def _init_publishers_subscribers(self):
        self.pub_cmd_left = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.pub_cmd_right = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)

        self.create_subscription(NavSatFix, 'wamv/sensors/gps/gps/fix', partial(self.gps_callback, wamv_id=1), 10)
        self.create_subscription(NavSatFix, 'wamv2/sensors/gps/gps/fix', partial(self.gps_callback, wamv_id=2), 10)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, 10)
        self.create_subscription(Float64, '/wamv/thrusters/left/thrust', self.left_thrust_callback, 10)
        self.create_subscription(Float64, '/wamv/thrusters/right/thrust', self.right_thrust_callback, 10)
        self.pub_k_value = self.create_publisher(Float64, '/wamv/k_value', 10)

        for i in range(1, 5):
            self.create_subscription(NavSatFix, f'/wamv{i}/sensors/gps/gps/fix', 
                                     lambda msg, i=i: self.gps_callback(msg, i), 10)

    def wgs84_to_utm_sydney(self, lon, lat):
        return self.utm_transformer.transform(lon, lat)
    
    def utm_to_robot_frame(self, utm_coord):
        if self.start is None:
            return utm_coord  # 또는 (0, 0)을 반환하거나 다른 적절한 처리
        return (utm_coord[0] - self.start[0], utm_coord[1] - self.start[1])
    
    def calculate_distance(self, point1, point2):
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def request_global_path(self):
        request = GetGlobalPath.Request()
        future = self.global_path_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.global_path_callback(future.result().path)
        else:
            self.get_logger().error('Failed to call get_global_path service')

    def global_path_callback(self, msg):
        self.global_path = [
            self.wgs84_to_utm_sydney(pose.pose.position.x, pose.pose.position.y)
            for pose in msg.poses
        ]
        self.get_logger().info(f"Received global path with {len(self.global_path)} waypoints")
        self.publish_path()

    def publish_path(self):
        start_publisher = self.create_publisher(Marker, '/rain/boat_dock/start_point', 10)
        goal_publisher = self.create_publisher(Marker, '/rain/boat_dock/goal_point', 10)
        path_publisher = self.create_publisher(Marker, '/rain/boat_dock/docking_path', 10)
        trajectory_publisher = self.create_publisher(Marker, '/rain/boat_dock/trajectory', 10)
        intermediate_goal_publisher = self.create_publisher(Marker, '/rain/boat_dock/waypoint', 10)
        current_pos_publisher = self.create_publisher(Marker, '/rain/boat_dock/current_position', 10)

        # 1. 시작점 마커
        if self.start is not None:
            start_marker = self.create_point_marker(self.start, "start_point", 0, [1.0, 1.0, 0.0], scale=1.0)  # 노란색
            start_publisher.publish(start_marker)  # 개별 토픽으로 발행

        # 2. 전역 경로 마커 
        if self.global_path:
            path_marker = Marker()
            path_marker.header.frame_id = "wamv/wamv/base_link"
            path_marker.header.stamp = self.get_clock().now().to_msg()
            path_marker.ns = "global_path"
            path_marker.id = 0
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            path_marker.pose.orientation.w = 1.0
            path_marker.scale.x = 0.2
            path_marker.color.a = 1.0
            path_marker.color.r = 0.0
            path_marker.color.g = 0.0
            path_marker.color.b = 1.0  # 파란색으로 변경

            for wp in self.global_path:
                point = Point()
                robot_frame_wp = self.utm_to_robot_frame(wp)
                point.x = robot_frame_wp[0]
                point.y = robot_frame_wp[1]
                point.z = 0.0
                path_marker.points.append(point)

            path_publisher.publish(path_marker)  


        # 실제 궤적 마커 (보라색으로 변경)
        if self.position_history:
            trajectory_marker = Marker()
            trajectory_marker.header.frame_id = "wamv/wamv/base_link"
            trajectory_marker.header.stamp = self.get_clock().now().to_msg()
            trajectory_marker.ns = "actual_trajectory"
            trajectory_marker.id = 1
            trajectory_marker.type = Marker.LINE_STRIP
            trajectory_marker.action = Marker.ADD
            trajectory_marker.pose.orientation.w = 1.0
            trajectory_marker.scale.x = 0.2
            trajectory_marker.color.a = 1.0
            trajectory_marker.color.r = 0.5  # 보라색으로 변경
            trajectory_marker.color.g = 0.0
            trajectory_marker.color.b = 0.5

            for pos in self.position_history:
                point = Point()
                robot_frame_pos = self.utm_to_robot_frame(pos)
                point.x = robot_frame_pos[0]
                point.y = robot_frame_pos[1]
                point.z = 0.0
                trajectory_marker.points.append(point)

            trajectory_publisher.publish(trajectory_marker)  

            # 목표점 마커
        if self.global_path:
            goal_marker = Marker()
            goal_marker.header.frame_id = "wamv/wamv/base_link"
            goal_marker.header.stamp = self.get_clock().now().to_msg()
            goal_marker.ns = "goal_point"
            goal_marker.id = 2
            goal_marker.type = Marker.SPHERE
            goal_marker.action = Marker.ADD
            goal_marker.pose.position.x = self.global_path[-1][0]
            goal_marker.pose.position.y = self.global_path[-1][1]
            goal_marker.pose.position.z = 0.0
            goal_marker.scale.x = 1.0
            goal_marker.scale.y = 1.0
            goal_marker.scale.z = 1.0
            goal_marker.color.a = 1.0
            goal_marker.color.r = 1.0
            goal_marker.color.g = 0.0
            goal_marker.color.b = 0.0

            goal_marker = self.create_point_marker(self.goal, "goal_point", 1, [1.0, 0.0, 0.0], scale=1.0)  # 빨간색
            goal_publisher.publish(goal_marker)  # 개별 토픽으로 발행

        # 중간 목표 마커 (주황색)
        if self.global_path and len(self.global_path) >= 20:
            intermediate_goal_marker = Marker()
            intermediate_goal_marker.header.frame_id = "wamv/wamv/base_link"
            intermediate_goal_marker.header.stamp = self.get_clock().now().to_msg()
            intermediate_goal_marker.ns = "intermediate_goal"
            intermediate_goal_marker.id = 3
            intermediate_goal_marker.type = Marker.SPHERE
            intermediate_goal_marker.action = Marker.ADD

            robot_frame_intermediate_goal = self.utm_to_robot_frame(self.global_path[-21])
            intermediate_goal_marker.pose.position.x = robot_frame_intermediate_goal[0]
            intermediate_goal_marker.pose.position.y = robot_frame_intermediate_goal[1]
            intermediate_goal_marker.pose.position.z = 0.0

            # 마커 크기 설정
            intermediate_goal_marker.scale.x = 1.0
            intermediate_goal_marker.scale.y = 1.0
            intermediate_goal_marker.scale.z = 1.0

            # 주황색 설정
            intermediate_goal_marker.color.a = 1.0  # 투명도
            intermediate_goal_marker.color.r = 1.0  # 빨강
            intermediate_goal_marker.color.g = 0.5  # 초록
            intermediate_goal_marker.color.b = 0.0  # 파랑

            intermediate_goal_publisher.publish(intermediate_goal_marker)

        if self.current_pos is not None:
            # 현재 WAMV 위치 마커
            current_pos_marker = self.create_point_marker(self.current_pos, "current_position", 4, [0.0, 1.0, 0.0], scale=1.0)  # 녹색, 크기 조정
            current_pos_publisher.publish(current_pos_marker)
        

    def create_point_marker(self, point, ns, id, color, scale=0.2):
        marker = Marker()
        marker.header.frame_id = "wamv/wamv/base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        robot_frame_point = self.utm_to_robot_frame(point)
        marker.pose.position.x = robot_frame_point[0]
        marker.pose.position.y = robot_frame_point[1]
        marker.pose.position.z = 0.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        return marker

   
    def publish_boat_direction(self):
        """
        무인 수상정의 현재 방향을 RViz에 화살표로 시각화하는 
        """
        self.boat_direction_publisher = self.create_publisher(Marker, '/rain/boat_dock/boat_direction', 10)

        arrow_marker = Marker()
        arrow_marker.header.frame_id = "wamv/wamv/base_link"  # 수상정의 좌표계
        arrow_marker.header.stamp = self.get_clock().now().to_msg()
        arrow_marker.ns = "boat_direction"
        arrow_marker.id = 0
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD

        # 화살표의 시작점 (현재 수상정의 위치)
        start_point = Point()
        robot_frame_current = self.utm_to_robot_frame(self.current_pos)
        start_point.x = robot_frame_current[0]
        start_point.y = robot_frame_current[1]
        start_point.z = 0.0

        # 화살표의 끝점 (수상정의 진행 방향을 나타냄)
        end_point = Point()
        arrow_length = 3.0  # 화살표의 길이 (단위: 미터)
        end_point.x = start_point.x + arrow_length * math.cos(self.current_yaw)
        end_point.y = start_point.y + arrow_length * math.sin(self.current_yaw)
        end_point.z = 0.0

        arrow_marker.points.append(start_point)
        arrow_marker.points.append(end_point)

        # 화살표 모양 설정
        arrow_marker.scale.x = 0.2  # 화살표의 샤프트(몸통) 두께
        arrow_marker.scale.y = 0.4  # 화살표의 헤드(머리) 두께
        arrow_marker.scale.z = 0.4  # 화살표의 헤드 길이
        arrow_marker.color.a = 1.0  # 불투명
        arrow_marker.color.r = 0.0  # 빨간색
        arrow_marker.color.g = 1.0  # 초록색
        arrow_marker.color.b = 0.0  # 파란색

        # 퍼블리셔로 화살표 퍼블리시
        self.boat_direction_publisher.publish(arrow_marker)

    def gps_callback(self, msg, wamv_id):
        utm_pos = self.wgs84_to_utm_sydney(msg.longitude, msg.latitude)

        if wamv_id == 1:
            if self.start is None:
                self.start = utm_pos
            self.current_pos = utm_pos
            self.position_history.append(self.current_pos)

            if hasattr(self, 'last_gps_time'):
                current_time = self.get_clock().now()
                dt = (current_time - self.last_gps_time).nanoseconds / 1e9
                distance = self.calculate_distance(self.current_pos, self.previous_pos)
                self.current_velocity = distance / dt

            self.previous_pos = self.current_pos
            self.last_gps_time = self.get_clock().now()
            
        elif wamv_id == 2:
            self.obstacles[wamv_id] = {'center': utm_pos, 'radius': 7.0}

        self.broadcast_tf()

    def broadcast_tf(self):  # 추가
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'wamv/wamv/base_link'
        robot_frame_pos = self.utm_to_robot_frame(self.current_pos)
        t.transform.translation.x = robot_frame_pos[0]
        t.transform.translation.y = robot_frame_pos[1]
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(self.current_yaw / 2)
        t.transform.rotation.w = math.cos(self.current_yaw / 2)
        self.tf_broadcaster.sendTransform(t)

    def imu_callback(self, msg):
        orientation_q = msg.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.current_yaw = euler_from_quaternion(orientation_list)   
    
    def get_target_point(self, current_pos):
        if not self.global_path:
            return None

        # Find the nearest point on the global path
        distances = [self.calculate_distance(wp, current_pos) for wp in self.global_path[self.current_waypoint_index:]]
        nearest_index = distances.index(min(distances)) + self.current_waypoint_index

        # Look ahead on the path
        for i in range(nearest_index, len(self.global_path)):
            if self.calculate_distance(self.global_path[i], current_pos) >= self.look_ahead_distance:
                self.current_waypoint_index = i
                return self.global_path[i]

        # If we've reached the end of the path, return the last point
        self.current_waypoint_index = len(self.global_path) - 1
        return self.global_path[-1]
    
    def left_thrust_callback(self, msg):
        self.left_thrust = msg.data
        self.set_k_value()

    def right_thrust_callback(self, msg):
        self.right_thrust = msg.data
        self.set_k_value()

    def set_k_value(self):
        avg_thrust = (self.left_thrust + self.right_thrust) / 2
        if self.current_velocity != 0:
            self.k_value = avg_thrust / self.current_velocity
            self.publish_k_value()  # Publish k value whenever it's updated
        else:
            self.get_logger().warning("Velocity is zero, cannot calculate k.")

    def publish_k_value(self):
        k_msg = Float64()
        k_msg.data = self.k_value
        self.pub_k_value.publish(k_msg)
        
    def get_angular_velocity(self, target, current_pos, current_yaw):
        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        target_yaw = math.atan2(dy, dx)

        # 현재 방향과 목표점 방향 사이의 각도 차이 
        alpha = target_yaw - current_yaw 
        K = 2 * math.sin(alpha) / self.look_ahead_distance
        return self.linear_velocity * K 
    
    def get_velocities(self, target, current_pos, current_yaw, distance_to_waypoint, distance_to_goal):
        w = self.get_angular_velocity(target, current_pos, current_yaw)
        v = self.linear_velocity

        if distance_to_waypoint < self.look_ahead_distance:
            if target == self.global_path[-1] and distance_to_goal < self.goal_threshold:
                return 0.0, 0.0, True # v, w, goal_reached        

        return v, w, False 
    
    def get_closest_point_on_path(self, current_pos):
        if not self.global_path:
            return None
        
        distances = [self.calculate_distance(wp, current_pos) for wp in self.global_path]
        min_distance_index = distances.index(min(distances))
        return self.global_path[min_distance_index]
    
    def get_thrusts(self, v, w):
        total_thrust = v * self.k_value
        thrust_difference = w * self.av_coefficient
        left_thrust = total_thrust + thrust_difference / 2
        right_thrust = total_thrust - thrust_difference / 2
        return self.limit_thrusts(left_thrust, right_thrust)
    
    def limit_thrusts(self, left_thrust, right_thrust):
        max_thrust = 300.0
        left_thrust = max(0.0, min(left_thrust, max_thrust))
        right_thrust = max(0.0, min(right_thrust, max_thrust))
        return left_thrust, right_thrust
    
    def publish_thrusts(self, left_thrust, right_thrust):
        self.pub_cmd_left.publish(Float64(data=left_thrust))
        self.pub_cmd_right.publish(Float64(data=right_thrust))    
         
    def control_loop(self):
        if self.start_time is None:
            self.start_time = self.get_clock().now()

        if not self.global_path:
            # 전역 경로가 없으면 요청
            self.request_global_path()
            return

        self.publish_boat_direction()
        # 먼저 중간 목표로 이동
        if not self.intermediate_goal_reached:
            intermediate_goal = self.global_path[-21]
            distance_to_goal = self.calculate_distance(intermediate_goal, self.current_pos)

            # 중간 목표 도달 확인
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info(f"Reached intermediate goal: {intermediate_goal}")
                self.stop_thrusters()  # 중간 목표 도달 시 정지
                self.intermediate_goal_reached = True  # 중간 목표 도달로 상태 변경
            else:
                target_point = self.get_target_point(self.current_pos)
                distance_to_waypoint = self.calculate_distance(target_point, self.current_pos)

                v, w, _ = self.get_velocities(target_point, self.current_pos, self.current_yaw, distance_to_waypoint, distance_to_goal)
                left_thrust, right_thrust = self.get_thrusts(v, w)
                self.publish_thrusts(left_thrust, right_thrust)

        
        else:
            # 최종 목표로 이동
            final_goal = self.global_path[-1]
            distance_to_final_goal = self.calculate_distance(final_goal, self.current_pos)

            # 최종 목표 도달 확인
            if distance_to_final_goal < self.goal_threshold:
                end_time = self.get_clock().now()
                elapsed_time = (end_time - self.start_time).nanoseconds / 1e9 # 초 단위로 변환
                self.get_logger().info(f"Goal reached. Total time: {elapsed_time:.2f} seconds")
                self.stop_thrusters()
                self.get_logger().info(f"Reached final goal. Distance: {distance_to_final_goal:.2f} m")
                self.goal_reached = True
                self.total_time = elapsed_time
                self.timer.cancel()  # 타이머 취소
                rclpy.shutdown()  # ROS 2 노드 종료
            else:
                target_point = self.get_target_point(self.current_pos)
                distance_to_waypoint = self.calculate_distance(target_point, self.current_pos)

                v, w, _ = self.get_velocities(target_point, self.current_pos, self.current_yaw, distance_to_waypoint, distance_to_final_goal)
                left_thrust, right_thrust = self.get_thrusts(v, w)
                self.publish_thrusts(left_thrust, right_thrust)

        
        self.publish_path()


    def stop_thrusters(self):
        self.pub_cmd_left.publish(Float64(data=0.0))
        self.pub_cmd_right.publish(Float64(data=0.0))

def main(args=None): 
    rclpy.init(args=args) 
    controller = DockingPathFollower()      
    try:
        rclpy.spin(controller)

    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()