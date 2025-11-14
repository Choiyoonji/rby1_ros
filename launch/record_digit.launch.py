from launch import LaunchDescription
from launch_ros.actions import Node
import os
from pathlib import Path

def generate_launch_description():
    home_dir = str(Path.home() / "rby1_data")
    return LaunchDescription([
        Node(
            package='rby1_ros',
            executable='tick_publisher',
            name='tick_publisher',
            output='screen',
            parameters=[
                {'task': 'digit_test_1114'},
                {'base_dir': home_dir}
            ]
        ),

        Node(
            package='rby1_ros',
            executable='realsense_record_node',
            name='right_wrist_D405',
            output='screen',
            parameters=[
                {'shm_name': 'right_wrist_D405'},
                {'camera_model': 'D405'},
                {'serial_number': '218622272411'}
            ]
        ),

        Node(
            package='rby1_ros',
            executable='realsense_record_node',
            name='external_D435I',
            output='screen',
            parameters=[
                {'shm_name': 'external_D435I'},
                {'camera_model': 'D435I'},
                {'serial_number': '233522076898'}
            ]
        ),

        Node(
            package='rby1_ros',
            executable='digit_record_node',
            name='right_hand_digit1',
            output='screen',
            parameters=[
                {'shm_name': 'right_hand_digit1'},
                {'serial_number': 'D20612'}
            ]
        ),

        Node(
            package='rby1_ros',
            executable='digit_record_node',
            name='right_hand_digit2',
            output='screen',
            parameters=[
                {'shm_name': 'right_hand_digit2'},
                {'serial_number': 'D20669'}
            ]
        ),
    ])
