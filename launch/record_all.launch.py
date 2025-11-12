from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rby1_ros',
            executable='tick_publisher',
            name='tick_publisher',
            output='screen',
            parameters=[
                {'task': 'test_task_001'},
                {'base_dir': '~/rby1_data'}
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
            name='left_wrist_D405',
            output='screen',
            parameters=[
                {'shm_name': 'left_wrist_D405'},
                {'camera_model': 'D405'},
                {'serial_number': '218622278157'}
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
            executable='zed_record_node',
            name='zed_record_node',
            output='screen'
        ),
    ])
