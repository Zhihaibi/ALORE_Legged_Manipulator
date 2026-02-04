cmake_minimum_required(VERSION 3.0.2)
project(front_end)


set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  roscpp
  sensor_msgs
  std_msgs
  visualization_msgs
  plan_env
)

find_package(cmake_modules REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES front_end
 CATKIN_DEPENDS cv_bridge geometry_msgs roscpp sensor_msgs std_msgs visualization_msgs plan_env
#  DEPENDS system_lib
)

add_library( front_end 
  src/jps_planner/graph_search.cpp
  src/jps_planner/jps_planner.cpp
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS} 
  ${PCL_INCLUDE_DIRS}
)
