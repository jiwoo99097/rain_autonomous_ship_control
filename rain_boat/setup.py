from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rain_boat'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), ['rain_boat/turtlebot_launch.py']), 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rain',
    maintainer_email='rain@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'boat_path = rain_boat.boat_path:main',
            'boat_dock = rain_boat.boat_dock:main',
            'turtlebot_path = rain_boat.turtlebot_path:main',
            'turtlebot_dock = rain_boat.turtlebot_dock:main',
            'goalpos_to_pos = rain_boat.goalpos_to_pos:main',
            'turtlebot_launch = rain_boat.turtlebot_launch:main',
            'trajectory_analysis = rain_boat.trajectory_analysis:main',
        ],
    },
)
