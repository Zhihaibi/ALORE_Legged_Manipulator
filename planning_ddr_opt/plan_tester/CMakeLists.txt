cmake_minimum_required(VERSION 3.0.2)
project(plan_tester)


set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  roscpp
  std_msgs
  visualization_msgs
  gazebo_msgs
  plan_env
  front_end
  back_end
  carstatemsgs
  mpc_controller
)

find_package(cmake_modules REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES plan_tester
 CATKIN_DEPENDS carstatemsgs gazebo_ros roscpp std_msgs visualization_msgs plan_env front_end back_end mpc_controller
#  CATKIN_DEPENDS gazebo_ros roscpp std_msgs visualization_msgs plan_env front_end
 DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS} 
  ${PCL_INCLUDE_DIRS}
)


add_executable(global_planning src/global_planning.cpp)

target_link_libraries(global_planning
  ${catkin_LIBRARIES}
)

# 添加新的验证节点可执行文件
add_executable(mission_validation_node src/mission_validation_node.cpp)

# 链接该节点到ROS库
target_link_libraries(mission_validation_node
  ${catkin_LIBRARIES}
)
