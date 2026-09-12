cmake_minimum_required(VERSION 3.5)
project(balance_robot_description)

find_package(ament_cmake REQUIRED)

# Cài đặt thư mục urdf và launch vào share
install(
  DIRECTORY
    urdf
    launch
    rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
