"""
Launch file chạy Balance Controller.

Sử dụng:
    ros2 launch balance_robot_controller controller.launch.py

Controller sẽ:
1. Đọc dữ liệu IMU từ /imu/data
2. Tính PID output
3. Publish vận tốc lên /cmd_vel

Yêu cầu: Gazebo phải đang chạy với robot đã spawn.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('balance_robot_controller')
    default_config_file = os.path.join(pkg_share, 'config', 'pid_params.yaml')

    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_file,
        description='Đường dẫn file YAML cấu hình PID'
    )

    balance_controller_node = Node(
        package='balance_robot_controller',
        executable='balance_controller',
        name='balance_controller',
        output='screen',
        parameters=[LaunchConfiguration('config_file'), {'use_sim_time': True}],
    )

    return LaunchDescription([
        config_arg,
        balance_controller_node,
    ])
