#!/bin/bash

# Install dependencies
sudo apt install libgflags-dev  ros-$ROS_DISTRO-image-geometry ros-$ROS_DISTRO-camera-info-manager ros-$ROS_DISTRO-image-transport ros-$ROS_DISTRO-image-publisher libgoogle-glog-dev libusb-1.0-0-dev libeigen3-dev

cd ~/turtlebot_ws

# Install libuvc
git clone https://github.com/libuvc/libuvc.git
cd libuvc
mkdir build && cd build
cmake .. && make -j4
sudo make install
sudo ldconfig

# Clone code
cd ~/turtlebot_ws/src

git clone https://github.com/orbbec/ros_astra_camera.git

cd ~/turtlebot_ws
catkin_make

# Install udev rules
source ./devel/setup.bash
roscd astra_camera
./scripts/create_udev_rules
sudo udevadm control --reload && sudo  udevadm trigger

cd ~/turtlebot_ws
. devel/setup.bash
