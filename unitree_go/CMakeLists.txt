cmake_minimum_required(VERSION 3.8)
project(unitree_go)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BmsCmd.msg"
  "msg/BmsState.msg"
  "msg/IMUState.msg"
  "msg/LowCmd.msg"
  "msg/LowState.msg"
  "msg/MotorCmd.msg"
  "msg/MotorState.msg"
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
