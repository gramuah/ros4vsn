#!/bin/bash

# Creating the directories where we will generate the workspace for the turtlebot packages
mkdir -p turtlebot_ws/src
cd turtlebot_ws/src

# Installing turtlebot packages and pasting in the src directory
git clone https://github.com/turtlebot/turtlebot.git
git clone https://github.com/turtlebot/turtlebot_msgs.git
git clone https://github.com/turtlebot/turtlebot_apps.git

#This clone is to avoid issues
git clone https://github.com/yujinrobot/yujin_ocs.git

#Exctract the directories that are interesting for us and we remove the leftovers
mv yujin_ocs/yocs_cmd_vel_mux yujin_ocs/yocs_controllers yujin_ocs/yocs_velocity_smoother ./
rm -rf yujin_ocs

#Adding the monitory package
git clone https://github.com/ros-drivers/linux_peripheral_interfaces.git
mv linux_peripheral_interfaces/laptop_battery_monitor ./
rm -rf linux_peripheral_interfaces

#Adding the Noetic branch of the kobuki git
git clone https://github.com/yujinrobot/kobuki.git

sudo apt install liborocos-kdl-dev -y

rosdep update
rosdep install --from-paths . --ignore-src -r -y

cd ..
catkin_make
