#!/bin/bash

# sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# Update package list
sudo apt-get update

# ROS Navigation Planner dependencies
sudo apt-get install -y ros-noetic-navigation

sudo apt-get install -y ros-noetic-laser-drivers
sudo apt-get install -y ros-noetic-depth-camera-drivers
sudo apt-get install ros-noetic-map-server
sudo apt-get install ros-noetic-amcl
sudo apt-get install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control
sudo apt-get install ros-noetic-rviz

echo "ROS Navigation Planner dependencies installed successfully."

# Update environment variables
source /opt/ros/noetic/setup.bash

# source ~/catkin_ws/devel/setup.bash

# Exit
exit 0

