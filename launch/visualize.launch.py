from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rby1_ros',
            executable='visualize_state_vs_command',
            name='visualize_state_vs_command',
            output='screen',
            parameters=[],
            remappings=[
                ('/rby1/state', '/rby1/state'),
                ('/control/command', '/control/command')
            ]
        )
    ])