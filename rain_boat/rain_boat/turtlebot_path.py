#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry
from global_path_service.srv import GetGlobalPath
from transforms3d.euler import quat2euler
import math
import random
import matplotlib.pyplot as plt
import logging

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
            
            smoothed_path.append((avg_x, avg_y))

        smoothed_path.append(path[-1])  # 끝점도 그대로 유지
        return smoothed_path

class PathPlanner(Node):
    def __init__(self):
        super().__init__('path_planner')
        
        self.start = None
        self.goal = None
        self.intermediate_goal = None  # 중간 목표 지점       
 
        self.path_generated = False
        self.global_path = None
 
        self.create_subscription(PoseStamped, '/rain/project_autonomous_ship/target_point', self.goal_callback, 10)
        self.create_subscription(PoseStamped, '/current_pose', self.position_callback, 10)
 
        self.client = self.create_client(GetGlobalPath, '/rain/docking_path')
       
        self.map_area = 50  
        self.dmax = 10  
        self.safety_margin = 3.0
    
    def initialize_parameters(self):
        """Initialize parameters and state variables."""
        self.start = None
        self.goal = None
        self.intermediate_goal = None
 
        self.map_area = 50  
        self.dmax = 10  
        self.safety_margin = 3.0
 
    def initialize_subscriptions(self):
        """Set up ROS topic subscriptions."""
        self.goal_subscription = self.create_subscription(
            PoseStamped,
            '/rain/project_autonomous_ship/target_point',
            self.goal_callback,
            10
        )
        self.position_subscription = self.create_subscription(
            PoseStamped,
            '/current_pose',
            self.position_callback,
            10
        )
 
    def initialize_service_client(self):
        """Initialize the service client for global path."""
        self.client = self.create_client(GetGlobalPath, '/rain/docking_path')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /rain/docking_path service...')
 
    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("BiasedRRTStar")
 
    def position_callback(self, msg):
        if self.start is None:
            self.current_position = (msg.pose.position.x, msg.pose.position.y)
            self.start = self.current_position
            self.get_logger().info(f"Start position received: x: {self.start[0]}, y: {self.start[1]}")
        
    def goal_callback(self, msg):
        """Handle incoming goal position messages."""
        if self.goal is None:
            self.set_goal(msg)
            self.generate_and_handle_path()
 
    def set_goal(self, msg):
        """Extract and set the goal and intermediate goal from the message."""
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        orientation_q = msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, goal_yaw = quat2euler(orientation_list)
 
        self.goal = (goal_x, goal_y)
        self.get_logger().info(f"Goal position received: x: {goal_x}, y: {goal_y}, yaw: {goal_yaw}")
 
        # Calculate intermediate goal based on orientation
        intermediate_goal_x = goal_x + 2.0 * math.cos(goal_yaw - math.pi / 2)
        intermediate_goal_y = goal_y + 2.0 * math.sin(goal_yaw - math.pi / 2)
        self.intermediate_goal = (intermediate_goal_x, intermediate_goal_y)
        self.get_logger().info(f"Intermediate goal set: x: {self.intermediate_goal[0]}, y: {self.intermediate_goal[1]}")
 
    def generate_and_handle_path(self):
        """Generate the path and handle the global path service call."""
        if not self.validate_path_generation_conditions():
            return
 
        path_to_intermediate, nodes_to_intermediate, parents_to_intermediate = self.generate_path_to_intermediate()
        if not path_to_intermediate:
            self.get_logger().warning("No path to intermediate goal found")
            return
 
        path_to_goal, nodes_to_goal, parents_to_goal = self.generate_path_to_goal()
        if not path_to_goal:
            self.get_logger().warning("No path to final goal found")
            return
 
        interpolated_path = self.interpolate_and_combine_paths(path_to_intermediate, path_to_goal)
        self.global_path = interpolated_path
        self.path_generated = True
 
        self.visualize_paths(
            interpolated_path,
            nodes_to_intermediate, parents_to_intermediate,
            nodes_to_goal, parents_to_goal
        )
 
        self.call_global_path_service()
 
    def validate_path_generation_conditions(self):
        """Ensure all necessary parameters are set before path generation."""
        if self.start is None or self.goal is None or self.intermediate_goal is None:
            self.get_logger().info("Waiting for start, goal, or intermediate positions")
            return False
        return True
 
    def generate_path_to_intermediate(self):
        """Generate path from start to intermediate goal."""
        biased_rrt_star = BiasedRRTStar(
            self.start,
            self.intermediate_goal,
            self.dmax,
            self.map_area,
            self.safety_margin
        )
        path, nodes, parents = biased_rrt_star.generate_path()
        if path:
            self.get_logger().info("Path to intermediate goal found.")
        else:
            self.get_logger().warning("No path to intermediate goal found")
        return path, nodes, parents
 
    def generate_path_to_goal(self):
        """Generate path from intermediate goal to final goal."""
        biased_rrt_star = BiasedRRTStar(
            self.intermediate_goal,
            self.goal,
            self.dmax,
            self.map_area,
            self.safety_margin
        )
        path, nodes, parents = biased_rrt_star.generate_path()
        if path:
            self.get_logger().info("Path to final goal found.")
        else:
            self.get_logger().warning("No path to final goal found")
        return path, nodes, parents
 
    def interpolate_and_combine_paths(self, path_to_intermediate, path_to_goal, num_points=10):
        """Interpolate both paths and combine them into a global path."""
        interpolated_path_to_intermediate = self.interpolate_path(path_to_intermediate, num_points)
        interpolated_path_to_goal = self.interpolate_path(path_to_goal, num_points)
        combined_path = interpolated_path_to_intermediate + interpolated_path_to_goal
        return combined_path
 
    def interpolate_path(self, path, num_points):
        """Interpolate a given path with a specified number of points between waypoints."""
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
 
    def visualize_paths(self, global_path, nodes_intermediate, parents_intermediate, nodes_goal, parents_goal):
        """Visualize the generated paths and trees."""
        plt.figure(figsize=(15, 15))
 
        # Plot the first tree (Start -> Intermediate Goal)
        for i, node in enumerate(nodes_intermediate):
            if i > 0:
                parent = nodes_intermediate[parents_intermediate[i]]
                plt.plot(
                    [node[0], parent[0]],
                    [node[1], parent[1]],
                    'b-', linewidth=0.5, alpha=0.3
                )
 
        # Plot the second tree (Intermediate Goal -> Goal)
        for i, node in enumerate(nodes_goal):
            if i > 0:
                parent = nodes_goal[parents_goal[i]]
                plt.plot(
                    [node[0], parent[0]],
                    [node[1], parent[1]],
                    'r-', linewidth=0.5, alpha=0.3
                )
 
        # Plot the global path
        global_path_x = [point[0] for point in global_path]
        global_path_y = [point[1] for point in global_path]
        plt.plot(global_path_x, global_path_y, 'g-', linewidth=2, label='Docking_Path')
 
        # Plot start, intermediate, and goal points
        plt.plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
        plt.plot(self.intermediate_goal[0], self.intermediate_goal[1], 'bo', markersize=10, label='Intermediate Goal')
        plt.plot(self.goal[0], self.goal[1], 'ro', markersize=10, label='Goal')
 
        plt.title('Path Plan for Docking')
        plt.xlabel('X coordinate (m)')
        plt.ylabel('Y coordinate (m)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig('Docking_Path.png')
        plt.close()
        self.get_logger().info("Path visualization saved as 'Docking_Path.png'.")
 
    def call_global_path_service(self):
        """Prepare and call the global path service with the generated path."""
        if not self.global_path:
            self.get_logger().warning("Global path is empty. Cannot call service.")
            return
 
        global_path_request = GetGlobalPath.Request()
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "ndt_map"  # Set the appropriate frame ID
 
        for waypoint in self.global_path:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = "ndt_map"
 
            pose = Pose()
            pose.position.x = waypoint[0]
            pose.position.y = waypoint[1]
            pose.position.z = 0.0
 
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0
 
            pose_stamped.pose = pose
            path_msg.poses.append(pose_stamped)
 
        global_path_request.path = path_msg
 
        self.get_logger().info("Calling global path service...")
        future = self.client.call_async(global_path_request)
        future.add_done_callback(self.handle_service_response)
 
    def handle_service_response(self, future):
        """Handle the response from the global path service."""
        try:
            response = future.result()
            self.get_logger().info("Global path service call successful.")
            # Process the response as needed
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
 
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