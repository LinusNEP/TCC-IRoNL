cmake_minimum_required(VERSION 3.0.2)

project(romr_bringup)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf
  tf2_ros
  # Uncomment when you add move_base-based navigation:
  # actionlib
  # move_base_msgs
)

catkin_package(
  CATKIN_DEPENDS
    roscpp rospy std_msgs sensor_msgs nav_msgs geometry_msgs tf tf2_ros
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# ---------------- Python nodes ----------------
catkin_install_python(PROGRAMS
  src/cmd_vel_watchdog.py
  src/romr_odom_node.py
  src/stability_test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# ---------------- C++ nodes ----------------
 add_executable(send_goals src/send_goals.cpp)
 target_link_libraries(send_goals ${catkin_LIBRARIES})
 add_dependencies(send_goals ${catkin_EXPORTED_TARGETS})

# ---------------- Non-code resources ----------------
# Copy launch files, URDF, maps, config, etc. into the install tree so
# $(find romr_bringup)/launch/... resolves after `catkin_make install`.
foreach(dir config launch maps meshes params urdf worlds)
  install(DIRECTORY ${dir}/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir}
    OPTIONAL)
endforeach()
