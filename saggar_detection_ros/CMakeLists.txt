cmake_minimum_required(VERSION 3.0.2)
project(saggar_detection_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  cv_bridge
  image_transport
  message_generation
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  SaggarDetection.msg
  SaggarArray.msg
  LayerInfo.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    cv_bridge
    image_transport
    message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Mark executable scripts for installation
catkin_install_python(PROGRAMS
  nodes/maskrcnn_detector_node.py
  nodes/edge_detector_node.py
  nodes/bag_player_node.py
  nodes/visualization_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark launch files for installation
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

## Mark config files for installation
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
