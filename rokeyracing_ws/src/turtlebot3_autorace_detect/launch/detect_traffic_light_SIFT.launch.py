from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Detect Traffic Light Node
        Node(
            package='turtlebot3_autorace_detect',
            executable='detect_traffic_light_SIFT',
            name='detect_traffic_light_SIFT',
            output='screen',
            remappings=[
                ('/detect/image_input', '/camera/image_compensated'),
                ('/detect/image_input/compressed', '/camera/image_compensated/compressed'),
                ('/detect/image_output', '/detect/image_traffic_sign'),
                ('/detect/image_output/compressed', '/detect/image_traffic_sign/compressed'),
            ]
        )
    ])
