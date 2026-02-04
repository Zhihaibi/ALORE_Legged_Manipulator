cmake_minimum_required(VERSION 3.0.2)
project(nmpc_controller)

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
 LIBRARIES nmpc_controller
 CATKIN_DEPENDS carstatemsgs message_generation nav_msgs roscpp std_msgs visualization_msgs back_end 
 DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS} 
  ${PCL_INCLUDE_DIRS}
)

# add_library( mpc_solver
#   include/nmpc_controller/quadrotor_mpc_codegen/acado_auxiliary_functions.c
#   include/nmpc_controller/quadrotor_mpc_codegen/acado_auxiliary_functions.h
#   include/nmpc_controller/quadrotor_mpc_codegen/acado_common.h
#   include/nmpc_controller/quadrotor_mpc_codegen/acado_integrator.c
#   include/nmpc_controller/quadrotor_mpc_codegen/acado_qpoases_interface.cpp
#   include/nmpc_controller/quadrotor_mpc_codegen/acado_qpoases_interface.hpp
#   include/nmpc_controller/quadrotor_mpc_codegen/acado_solver.c
# )

add_executable(nmpc
  src/mpc_node.cpp
  src/mpc.cpp
  src/mpc_wrapper.cpp

  externals/qpoases/SRC/Bounds.cpp
  externals/qpoases/SRC/Constraints.cpp
  externals/qpoases/SRC/CyclingManager.cpp
  externals/qpoases/SRC/Indexlist.cpp
  externals/qpoases/SRC/MessageHandling.cpp
  externals/qpoases/SRC/QProblem.cpp
  externals/qpoases/SRC/QProblemB.cpp
  externals/qpoases/SRC/SubjectTo.cpp
  externals/qpoases/SRC/Utils.cpp
  externals/qpoases/SRC/EXTRAS/SolutionAnalysis.cpp
  UAV_CAR_model/build/quadrotor_mpc_codegen/acado_qpoases_interface.cpp
  UAV_CAR_model/build/quadrotor_mpc_codegen/acado_integrator.c
  UAV_CAR_model/build/quadrotor_mpc_codegen/acado_solver.c
  UAV_CAR_model/build/quadrotor_mpc_codegen/acado_auxiliary_functions.c
  
)

# add_library(nmpc_controller 
#   src/mpc_node.cpp
#   src/mpc.cpp 
# )

target_include_directories(nmpc PUBLIC
  include/nmpc_controller
  include 
  ${EIGEN3_INCLUDE_DIR}
  SYSTEM
  ${catkin_INCLUDE_DIRS} 
  # MPC solver
  UAV_CAR_model/build/quadrotor_mpc_codegen
  externals/qpoases
  externals/qpoases/INCLUDE
  externals/qpoases/SRC
  ${CERES_INCLUDE_DIRS}
)

target_link_libraries(nmpc
  OsqpEigen::OsqpEigen
  ${catkin_LIBRARIES}
  libosqp.so  
  libOsqpEigen.so 
  ${CERES_LIBRARIES}
  # libqdldl.so 
)