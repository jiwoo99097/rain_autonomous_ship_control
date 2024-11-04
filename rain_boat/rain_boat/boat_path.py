#!/usr/bin/env python3

# ros2 launch vrx_gz competition.launch.py world:=sydney_regatta

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from global_path_service.srv import GetGlobalPath
import math
import random
import matplotlib.pyplot as plt
import logging
import pyproj as proj
import numpy as np
    
class BiasedRRTStar:
    def __init__(self, start, goal, dmax, map_area, safety_margin=2.0):
        self.start = start
        self.goal = goal
        self.dmax = dmax
        self.map_area = map_area
        self.safety_margin = safety_margin
        self.nodes = [start]
        self.parents = [0]
        self.costs = [0]
        self.near_radius = 50.0  # Radius for considering nearby nodes
        self.goal_bias = 0.4  # 목표 지점으로의 편향 확률
        self.goal_zone_radius = self.distance(start, goal) * 0.2  # 목표 주변 20% 영역을 목표 존으로 설정
        self.heuristic_bias = 0.6  # 목표 방향으로의 편향 강도
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("BiasedRRTStar")

    def generate_random_point(self):
        rand = random.random()
        if rand < self.goal_bias:
            # 목표점 근처의 랜덤한 점 선택
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, self.goal_zone_radius)
            x = self.goal[0] + distance * math.cos(angle)
            y = self.goal[1] + distance * math.sin(angle)
            return (x, y)
        elif rand < self.goal_bias + self.heuristic_bias:
            # 시작점에서 목표점 방향으로의 원뿔 모양 영역 내에서 점 선택
            angle_to_goal = math.atan2(self.goal[1] - self.start[1], self.goal[0] - self.start[0])
            angle_variance = math.pi / 4  # 45도 변화 허용
            random_angle = random.uniform(angle_to_goal - angle_variance, angle_to_goal + angle_variance)
            distance = random.uniform(0, self.distance(self.start, self.goal))
            x = self.start[0] + distance * math.cos(random_angle)
            y = self.start[1] + distance * math.sin(random_angle)
            return (x, y)
        else:
            # 완전히 랜덤한 점 선택 (기존 방식)
            return (random.uniform(-self.map_area, self.map_area),
                    random.uniform(-self.map_area, self.map_area))

    def nearest_node(self, point):
        distances = [self.distance(n, point) for n in self.nodes]
        return self.nodes[distances.index(min(distances))]

    def distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def step(self, p1, p2):
        if self.distance(p1, p2) < self.dmax:
            return p2
        else:
            theta = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
            return (p1[0] + self.dmax*math.cos(theta),
                    p1[1] + self.dmax*math.sin(theta))

    def closest_point_on_segment(self, p1, p2, p):
        x1, y1 = p1
        x2, y2 = p2
        px, py = p
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return p1
        t = max(0, min(1, ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)))
        return (x1 + t*dx, y1 + t*dy)

    def near_nodes(self, point):
        return [i for i, node in enumerate(self.nodes) if self.distance(node, point) < self.near_radius]

    def generate_path(self):
        self.logger.info("Starting Biased-RRT* path generation")
        iterations = 300  # 고정된 iteration 횟수
        best_path = None
        best_cost = float('inf')

        for iteration in range(iterations):
            if iteration % 50 == 0:
                self.logger.info(f"Iteration {iteration}")
            
            rnd_point = self.generate_random_point()
            nearest = self.nearest_node(rnd_point)
            new_point = self.step(nearest, rnd_point)
            
            near_inds = self.near_nodes(new_point)
            new_node_ind = self.choose_parent(new_point, near_inds)
               
            if new_node_ind is not None:
                self.nodes.append(new_point)
                self.rewire(len(self.nodes) - 1, near_inds)
                
                if self.distance(new_point, self.goal) < self.dmax:
                    self.nodes.append(self.goal)
                    self.parents.append(len(self.nodes) - 2)
                    path_cost = self.costs[-1] + self.distance(new_point, self.goal)
                    self.costs.append(path_cost)
                    
                    if path_cost < best_cost:
                        best_cost = path_cost
                        best_path = self.extract_path()
                        self.logger.info(f"New best path found at iteration {iteration + 1} with cost {best_cost}")

        self.logger.info(f"Completed {iterations} iterations")
        
        if best_path:
            self.logger.info(f"Best path found with cost: {best_cost}")
            smoothed_path = self.path_smoothen(best_path)
            return smoothed_path, self.nodes, self.parents
        else:
            self.logger.warning("No path to goal found")
            return None, self.nodes, self.parents

    def choose_parent(self, new_point, near_inds):
        if not near_inds:
            return None

        costs = [self.costs[i] + self.distance(self.nodes[i], new_point) for i in near_inds]
        min_cost_ind = near_inds[costs.index(min(costs))]

        self.parents.append(min_cost_ind)
        self.costs.append(min(costs))
        return min_cost_ind

    def rewire(self, new_node_ind, near_inds):
        for i in near_inds:
            near_node = self.nodes[i]
            edge_node = self.nodes[new_node_ind]
            
            cost = self.costs[new_node_ind] + self.distance(near_node, edge_node)
            
            if cost < self.costs[i]:
                self.parents[i] = new_node_ind
                self.costs[i] = cost

    def extract_path(self):
        if self.nodes[-1] != self.goal:
            return None
        
        path = [self.goal]
        parent_index = len(self.nodes) - 2
        while parent_index != 0:
            path.append(self.nodes[parent_index])
            parent_index = self.parents[parent_index]
        path.append(self.start)
        return list(reversed(path))
    
    def path_smoothen(self, path, window_size=5):
        if len(path) <= window_size:
            return path

        smoothed_path = [path[0]]  # 시작점은 그대로 유지

        for i in range(1, len(path) - 1):
            start = max(0, i - window_size // 2)
            end = min(len(path), i + window_size // 2 + 1)
            window = path[start:end]
            
            avg_x = sum(p[0] for p in window) / len(window)
            avg_y = sum(p[1] for p in window) / len(window)
            
            # 장애물과의 충돌 검사
            if all(self.distance((avg_x, avg_y), obs[:2]) > obs[2] + self.safety_margin for obs in self.obstacles):
                smoothed_path.append((avg_x, avg_y))
            else:
                smoothed_path.append(path[i])

        smoothed_path.append(path[-1])  # 끝점도 그대로 유지
        return smoothed_path

class PathPlanner(Node):
    def __init__(self):
        super().__init__('path_planner')
        
        self.start = None
        self.goal = None
        self.intermediate_goal = None # 중간 목표 지점 
        
        self.origin_lat = None
        self.origin_lon = None
        self.cust = None
        # self.obstacles = []
        self.path_generated = False
        self.global_path = None
        self.offset_x = None
        self.offset_y = None

        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', self.wamv_gps_callback, 10)

        self.srv = self.create_service(GetGlobalPath, '/rain/docking_path', self.get_global_path_callback)
        
        self.map_area = 50  
        self.dmax = 10  
        self.safety_margin = 3.0

        self.goal_lat = -33.72219231290662
        self.goal_lon = 150.67352145568518

        self.intermediate_goal_lat = -33.722213717246916
        self.intermediate_goal_lon = 150.67385537274836

    def get_global_path_callback(self, request, response):
        if not self.path_generated:
            self.generate_path()
        
        if self.global_path:
            response.path = self.create_path_msg(self.global_path)
        else:
            self.get_logger().warning("No path generated to return")

        return response
    
    def get_local_coord(self, lat, lon):
        WORLD_POLAR_M = 6356752.3142
        WORLD_EQUATORIAL_M = 6378137.0

        eccentricity = math.acos(WORLD_POLAR_M/WORLD_EQUATORIAL_M)        
        n_prime = 1/(math.sqrt(1 - math.pow(math.sin(math.radians(float(lat))),2.0)*math.pow(math.sin(eccentricity), 2.0)))        
        m = WORLD_EQUATORIAL_M * math.pow(math.cos(eccentricity), 2.0) * math.pow(n_prime, 3.0)        
        n = WORLD_EQUATORIAL_M * n_prime

        diffLon = float(lon) - float(self.origin_lon)
        diffLat = float(lat) - float(self.origin_lat)

        surfdistLon = math.pi /180.0 * math.cos(math.radians(float(lat))) * n
        surfdistLat = math.pi/180.00 * m

        x = diffLon * surfdistLon
        y = diffLat * surfdistLat

        return x, y

    def XY_to_LatLon(self, x_point, y_point):
        lon, lat = self.cust(x_point, y_point, inverse=True)
        return lon, lat

    def wamv_gps_callback(self, msg):
        if self.origin_lat is None:
            self.origin_lat = msg.latitude
            self.origin_lon = msg.longitude
            self.cust = proj.Proj(f"+proj=aeqd +lat_0={self.origin_lat} +lon_0={self.origin_lon} +datum=WGS84 +units=m")
        
        x, y = self.get_local_coord(msg.latitude, msg.longitude)

        if self.start is None:
            self.start = (0, 0)
            self.offset_x, self.offset_y = x, y
            
            goal_x, goal_y = self.get_local_coord(self.goal_lat, self.goal_lon)
            self.goal = (goal_x - self.offset_x, goal_y - self.offset_y)
            
            self.generate_path()

    def create_path_msg(self, global_path):
        path_msg = Path()
        now = self.get_clock().now().to_msg()
        path_msg.header.stamp = now
        path_msg.header.frame_id = "global_map"

        for x, y in global_path:
            pose = PoseStamped()
            x_global = x + self.offset_x
            y_global = y + self.offset_y
            lon, lat = self.XY_to_LatLon(x_global, y_global)
            pose.pose.position.x = lon
            pose.pose.position.y = lat
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

        return path_msg
    
    def interpolate_path(self, path, num_points):
        interpolated_path = []
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i+1]
            for j in range(num_points):
                t = j / num_points
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                interpolated_path.append((x, y))
        interpolated_path.append(path[-1])
        return interpolated_path
    
    def generate_path(self):
        if self.path_generated:
            return

        if self.start is None or self.goal is None:
            self.get_logger().info("Waiting for start and goal positions")
            return

        # 중간 목표의 좌표를 구합니다.
        intermediate_goal_x, intermediate_goal_y = self.get_local_coord(self.intermediate_goal_lat, self.intermediate_goal_lon)
        self.intermediate_goal = (intermediate_goal_x - self.offset_x, intermediate_goal_y - self.offset_y)

        # 중간 목표까지의 경로를 먼저 생성합니다.
        biased_rrt_star = BiasedRRTStar(self.start, self.intermediate_goal, self.dmax, self.map_area, self.safety_margin)
        path_to_intermediate, nodes_to_intermediate, parents_to_intermediate = biased_rrt_star.generate_path()

        if path_to_intermediate:
            self.get_logger().info("Path to intermediate goal found.")
        else:
            self.get_logger().warning("No path to intermediate goal found")
            return

        # 중간 목표에서 최종 목표까지의 경로를 생성합니다.
        biased_rrt_star = BiasedRRTStar(self.intermediate_goal, self.goal, self.dmax, self.map_area, self.safety_margin)
        path_to_goal, nodes_to_goal, parents_to_goal = biased_rrt_star.generate_path()

        if path_to_goal:
            self.get_logger().info("Path to final goal found.")
        else:
            self.get_logger().warning("No path to final goal found")
            return

        # 중간 목표까지의 경로를 보간합니다.
        interpolated_path_to_intermediate = self.interpolate_path(path_to_intermediate, 10)

        # 중간 목표에서 최종 목표까지의 경로를 보간합니다.
        interpolated_path_to_goal = self.interpolate_path(path_to_goal, 10)

        # 두 보간된 경로를 하나로 합칩니다.
        self.global_path = interpolated_path_to_intermediate + interpolated_path_to_goal
        self.path_generated = True

        # 중간 목표가 global_path에서 몇 번째인지 계산
        intermediate_goal_index = len(interpolated_path_to_intermediate) - 1

        self.get_logger().info(f"Global path with intermediate goal found. Intermediate goal is at index {intermediate_goal_index}.")

        
        # 첫 번째 트리와 두 번째 트리를 모두 시각화합니다.
        self.visualize(self.global_path, nodes_to_intermediate, parents_to_intermediate, nodes_to_goal, parents_to_goal)

    def visualize(self, global_path, nodes_intermediate, parents_intermediate, nodes_goal, parents_goal):
        plt.figure(figsize=(15, 15))

        # 첫 번째 트리 (Start -> Intermediate Goal)
        for i, node in enumerate(nodes_intermediate):
            if i > 0:
                parent = nodes_intermediate[parents_intermediate[i]]
                plt.plot([node[0], parent[0]], [node[1], parent[1]], 'b-', linewidth=0.5, alpha=0.3)

        # 두 번째 트리 (Intermediate Goal -> Goal)
        for i, node in enumerate(nodes_goal):
            if i > 0:
                parent = nodes_goal[parents_goal[i]]
                plt.plot([node[0], parent[0]], [node[1], parent[1]], 'r-', linewidth=0.5, alpha=0.3)  # 다른 색상 사용

        # 전체 경로 시각화
        global_path_x = [point[0] for point in global_path]
        global_path_y = [point[1] for point in global_path]
        plt.plot(global_path_x, global_path_y, 'g-', linewidth=2, label='Docking_Path')

        plt.plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
        plt.plot(self.intermediate_goal[0], self.intermediate_goal[1], 'bo', markersize=10, label='Intermediate Goal')  # 중간 목표
        plt.plot(self.goal[0], self.goal[1], 'ro', markersize=10, label='Goal')

        plt.title('Path Plan for Docking')
        plt.xlabel('X coordinate (m)')
        plt.ylabel('Y coordinate (m)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig('Docking_Path.png')
        plt.close()

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlanner()
    
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()