cmake_minimum_required(VERSION 3.0.2)
project(simulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-O3 -Wall -fPIC)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(catkin REQUIRED COMPONENTS
  carstatemsgs
  geometry_msgs
  roscpp
  std_msgs
  visualization_msgs
  octomap_ros
  octomap_msgs
)

find_package(cmake_modules REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(octomap REQUIRED)   # 这里是 C++ 库，不是 ROS 包

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES simulator
  CATKIN_DEPENDS carstatemsgs geometry_msgs roscpp std_msgs visualization_msgs octomap_ros octomap_msgs
  DEPENDS octomap   # 非 ROS 包用 DEPENDS
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
)

include_directories("/usr/include/eigen3")

add_executable(simulator src/simulator.cpp)
target_link_libraries(simulator
  ${catkin_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
)

add_executable(global_map_node src/global_map.cpp)
target_link_libraries(global_map_node
  ${catkin_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

