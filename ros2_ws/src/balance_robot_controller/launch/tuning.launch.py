"""
Launch file chạy PID Auto-Tuner.

Sử dụng:
    ros2 launch balance_robot_controller tuning.launch.py

Tuner sẽ:
1. Ổn định robot
2. Áp dụng relay feedback
3. Đo dao động → tính Ku, Tu
4. Tính Kp, Ki, Kd theo Ziegler-Nichols
5. Kiểm nghiệm kết quả
6. Lưu file ~/tuned_pid_params.yaml

Yêu cầu: Gazebo phải đang chạy với robot đã spawn.

LƯU Ý: KHÔNG chạy cùng lúc với controller.launch.py
        vì cả 2 đều publish lên /cmd_vel!
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('balance_robot_controller')
    config_file = os.path.join(pkg_share, 'config', 'pid_params.yaml')

    pid_tuner_node = Node(
        package='balance_robot_controller',
        executable='pid_tuner',
        name='pid_tuner',
        output='screen',
        parameters=[config_file, {'use_sim_time': True}],
    )

    return LaunchDescription([
        pid_tuner_node,
    ])
