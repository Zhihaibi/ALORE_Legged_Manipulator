cmake_minimum_required(VERSION 3.0.2)
project(mpc_controller)

set(CMAKE_BUILD_TYPE "Release")
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(catkin REQUIRED COMPONENTS
  carstatemsgs
  nav_msgs
  roscpp
  std_msgs
  visualization_msgs
  back_end
)

find_package(OsqpEigen REQUIRED)
find_package(cmake_modules REQUIRED)
find_package(PCL REQUIRED)

include_directories(/usr/local/include/osqp)

# add_message_files(FILES
#   Polynome.msg
# )

# generate_messages(DEPENDENCIES
# 	std_msgs
#   geometry_msgs
# )

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES mpc_controller
 CATKIN_DEPENDS carstatemsgs message_generation nav_msgs roscpp std_msgs visualization_msgs back_end 
 DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS} 
  ${PCL_INCLUDE_DIRS}
)


add_executable(mpc
  src/mpc_node.cpp
  src/mpc.cpp
)

add_library(mpc_controller 
  src/mpc_node.cpp
  src/mpc.cpp 
)

target_link_libraries(mpc
  OsqpEigen::OsqpEigen
  ${catkin_LIBRARIES}
  libosqp.so  
  libOsqpEigen.so 
  # libqdldl.so 
)