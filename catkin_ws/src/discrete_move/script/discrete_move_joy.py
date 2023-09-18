#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
from move_robot import MoveRobot
from sensor_msgs.msg import Joy


def callback(data):
    global semaforo, key
    lista = [0, 1, 2, 3]
    if semaforo == 0:
        index = [i for i, e in enumerate(data.buttons) if e != 0]
        if index != [] and index[0] in lista:
            semaforo = 1
            key = index[0]


if __name__ == '__main__':
    global semaforo, key
    semaforo = 1

    robot = MoveRobot()

    # Create a ROS node with a name for our program
    rospy.init_node("discrete_move_joy", log_level=rospy.INFO)
    # Define a callback to stop the robot when we interrupt the program (CTRL-C)
    rospy.on_shutdown(robot.stop_robot)

    rospy.Subscriber("/joy", Joy, callback)

    command = {
        0: robot.move_backward,
        1: robot.turn_right,
        2: robot.turn_left,
        3: robot.move_forward
    }

    while not robot.position_robot and not rospy.is_shutdown():
        time.sleep(0.1)

    semaforo = 0
    while not rospy.is_shutdown():
        if semaforo == 1:
            command[key]()
            semaforo = 0
