cmake_minimum_required(VERSION 3.0.2)
project(discrete_move)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  std_msgs
)

## Generate services in the 'srv' folder
 add_service_files(
   FILES
   DiscreteServer.srv
 )

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
 )

## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
)
