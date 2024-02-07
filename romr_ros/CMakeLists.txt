cmake_minimum_required(VERSION 2.8.3)

project(romr_ros)

## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(PCL REQUIRED)
#include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  actionlib
  move_base_msgs
  tf
  cv_bridge
  sensor_msgs
  pcl_conversions
  pcl_ros
)

find_package( OpenCV REQUIRED )

#find_package(catkin REQUIRED)

catkin_package()

include_directories(
 include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
   src/send_waypoints.py
   src/send_goals.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
 
 INCLUDE_DIRECTORIES(/usr/local/lib)
LINK_DIRECTORIES(/usr/local/lib)

add_executable(send_goals src/send_goals.cpp)
target_link_libraries(send_goals ${catkin_LIBRARIES})

find_package(roslaunch)

foreach(dir config gazebo launch maps meshes params src urdf worlds)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
