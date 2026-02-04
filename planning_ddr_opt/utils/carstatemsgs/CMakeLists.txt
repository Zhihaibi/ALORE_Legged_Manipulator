cmake_minimum_required(VERSION 3.0.2)
project(carstatemsgs)


# set(CMAKE_CXX_FLAGS "-std=c++14")
# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  message_generation
  std_msgs
  geometry_msgs
)

add_message_files(
  FILES
  CarState.msg
  KinematicState.msg
  CarControl.msg
  SimulatedCarState.msg
  Polynome.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

# find_package(cmake_modules REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES carstatemsgs
 CATKIN_DEPENDS roscpp message_generation std_msgs geometry_msgs
)

