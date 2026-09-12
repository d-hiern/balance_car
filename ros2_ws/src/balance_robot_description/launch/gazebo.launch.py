"""
Launch file để mô phỏng robot trong Gazebo.
Sử dụng: ros2 launch balance_robot_description gazebo.launch.py

Robot sẽ được spawn ở vị trí z=0.055m để bánh xe chạm mặt đất.

Topics quan trọng:
  - /imu/data      : Dữ liệu cảm biến IMU (sensor_msgs/msg/Imu)
  - /cmd_vel       : Lệnh vận tốc (geometry_msgs/msg/Twist)
  - /odom          : Odometry (nav_msgs/msg/Odometry)
  - /joint_states  : Trạng thái khớp (sensor_msgs/msg/JointState)
"""
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro


def generate_launch_description():
    pkg_share = get_package_share_directory('balance_robot_description')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')
    xacro_file = os.path.join(pkg_share, 'urdf', 'balance_robot.urdf.xacro')

    robot_description_config = xacro.process_file(xacro_file)
    robot_description = robot_description_config.toxml()

    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Sử dụng thời gian mô phỏng từ Gazebo'
    )

    world = DeclareLaunchArgument(
        'world',
        default_value='',
        description='Đường dẫn đến file world Gazebo (trống = empty world)'
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'verbose': 'true',
            'world': LaunchConfiguration('world'),
        }.items(),
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
    )

    # Spawn robot: z=0.055 = ground_clearance
    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_balance_robot',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'balance_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.055',
        ],
    )

    return LaunchDescription([
        use_sim_time,
        world,
        gazebo,
        robot_state_publisher_node,
        spawn_entity_node,
    ])
