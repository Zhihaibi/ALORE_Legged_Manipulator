cmake_minimum_required(VERSION 2.8.3)
project(laser_simulator)
SET(CMAKE_BUILD_TYPE Release) 
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -use_fast_math")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_ros
  sensor_msgs
  geometry_msgs
  nav_msgs
  carstatemsgs
)

catkin_package(
  DEPENDS Eigen
  CATKIN_DEPENDS roscpp pcl_ros carstatemsgs
)

find_package(PCL REQUIRED)

include_directories(
  SYSTEM 
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(
  laser_sim_node
  src/laser_sim_node.cpp
)

target_link_libraries( 
  laser_sim_node
  ${catkin_LIBRARIES}
)

add_executable(
  transmit_odom_node
  src/transmit_odom.cpp
)

target_link_libraries( 
  transmit_odom_node
  ${catkin_LIBRARIES}
)

