from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rain_boat',  
            executable='turtlebot_path',   
            name='PathPlanner',     
            output='screen'
        ),
        Node(
            package='rain_boat',  
            executable='goalpos_to_pos',  
            name='GoalPostoPoseStamped',  
            output='screen'
        ),
        Node(
            package='rain_boat', 
            executable='turtlebot_dock',  
            name='PurePursuitPathController',  
            output='screen'
        )
    ])
