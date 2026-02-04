cmake_minimum_required(VERSION 3.0.2)
project(icrekf)

set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(catkin REQUIRED COMPONENTS
  carstatemsgs
  geometry_msgs
  roscpp
  std_msgs
  visualization_msgs
)

find_package(cmake_modules REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES icrekf
 CATKIN_DEPENDS carstatemsgs geometry_msgs roscpp std_msgs visualization_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)
include_directories("/usr/include/eigen3")

add_executable(icrekf 
  src/icrekf_node.cpp
  src/icrekf.cpp
)

target_link_libraries(icrekf
  ${catkin_LIBRARIES}
)