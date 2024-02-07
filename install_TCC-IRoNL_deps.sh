#!/bin/bash
#
# This script is used to install all the dependencies of the TCC-IRoNLEnv
# Version: 1.0
# Author: Nwankwo Linus @LinuxNEP
# Date: 05.10.2023

# Create and activate a virtual environment for the TCC-IRoNLEnv

# Check if the virtual environment directory already exists
if [ ! -d "TCC-IRoNLEnv" ]; then
    echo "Creating a virtual environment..."
    python3 -m venv TCC-IRoNLEnv
else
    echo "Virtual environment already exists. Skipping creation..."
fi

# Activate the virtual environment
source TCC-IRoNLEnv/bin/activate

# Ensure that ROS environment variables are still sourced correctly
source /opt/ros/noetic/setup.bash

# Upgrade pip inside the virtual environment (optional but recommended)
pip install --upgrade pip

# Install the packages that are required for the TCC-IRoNLEnv 
# inside the virtual environment
sudo apt update
sudo apt upgrade
sudo apt install python3
sudo apt install python3-pip
sudo apt-get install python3-dev
sudo apt-get install python3-tk
pip3 install rospkg
pip3 install defusedxml
pip3 install setuptools_rust
pip3 install --upgrade pip setuptools
pip3 install transformers
pip3 install torch #This requires a very strong internet connection
pip3 install netifaces
pip3 install Pillow
pip3 install torchvision
pip3 install ftfy
pip3 install opencv-python

# You should be in the root of your ROS workspace:
rosdep install --from-paths src --ignore-src -r -y

echo "All done! Make sure to activate the virtual environment and source ROS environment variables whenever you work on TCC-IRoNLEnv."

# Update and activate the virtual environment
sudo apt update
source TCC-IRoNLEnv/bin/activate

