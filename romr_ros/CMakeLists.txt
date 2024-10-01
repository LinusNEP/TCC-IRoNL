cmake_minimum_required(VERSION 2.8.3)

project(romr_ros)

## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  actionlib
  move_base_msgs
  tf
)

find_package( OpenCV REQUIRED )


#find_package(catkin REQUIRED)

catkin_package()

include_directories(
 include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
   # src/send_waypoints.py
   # src/odom_data.py
   src/stability_test.py
   
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
 
 INCLUDE_DIRECTORIES(/usr/local/lib)
LINK_DIRECTORIES(/usr/local/lib)

add_executable(send_goals src/send_goals.cpp)
target_link_libraries(send_goals ${catkin_LIBRARIES})

add_executable(odomtorobottfgenerator src/odomtorobottfgenerator.cpp)
target_link_libraries(odomtorobottfgenerator ${catkin_LIBRARIES})

add_executable(imuOrientToCmdVelTranslater src/imuOrientToCmdVelTranslater.cpp)
target_link_libraries(imuOrientToCmdVelTranslater ${catkin_LIBRARIES})
add_dependencies(imuOrientToCmdVelTranslater ${catkin_EXPORTED_TARGETS})

# add_executable(detectAndDecide src/detectAndDecide.cpp)
# target_link_libraries(detectAndDecide 
# ${catkin_LIBRARIES}
# ${OpenCV_LIBRARIES}
# )
find_package(roslaunch)

foreach(dir config gazebo launch maps meshes params src urdf worlds)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
