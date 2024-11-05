# Autonomous Ship Control Package Installation Guide
This guide provides instructions for launching necessary packages for Autonomous Ship Control Package, including ROS2 Humble installation. 

***
## Table of Contents
1. [ROS2 Humble Installation](#1.ROS2-Humble-Installation)
2. [Autonomous Ship Controll Package Installation](#2.Autonomous-Ship-Control-Package-Installation)
3. [Usage](#3.Usage)

***
## 1.ROS2 Humble Installation 

### You can install ROS2 Humble by referring to the link below.
```
https://docs.ros.org/en/humble/Installation.html
```
***

## 2.Autonomous Ship Control Pacakgae Installation 
 
 ```python
git clone https://github.com/jiwoo99097/rain_autonomous_ship_control.git
```
***

## 3.Usage
### Step 1:
### Install ROS2 Humble

### Step 2:
### Clone the repository to your system 
```
git clone https://github.com/jiwoo99097/rain_autonomous_ship_control.git
cd your_workspace
```

### Step 3:
### Build the package and source the Setup Script
```
cd your_workspace

source opt/ros/humble/setup.bash
colcon build 
source install/setup.bash
```

### Step 4:
### with ros2 launch:
```
ros2 launch rain_boat turtlebot_launch.py
```
### ROS Topic should be like
```
/cmd_vel
/current_pose
/parameter_events
/rain/boat_dock/boat_direction
/rain/boat_dock/current_position
/rain/boat_dock/global_path
/rain/boat_dock/goal_point
/rain/boat_dock/intermediate_goal
/rain/boat_dock/start_point
/rain/boat_dock/trajectory
/rain/goal_position
/rain/project_autonomous_ship/target_point
```
