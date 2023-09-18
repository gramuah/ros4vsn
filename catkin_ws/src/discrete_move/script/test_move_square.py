#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
from move_robot import MoveRobot


if __name__ == '__main__':

    robot = MoveRobot()

    rospy.init_node("test_move_square", log_level=rospy.INFO)
    rospy.on_shutdown(robot.stop_robot)

    while not robot.position_robot and not rospy.is_shutdown():
        time.sleep(0.1)

    robot.move_forward()
    robot.turn_left()

    robot.move_forward()
    robot.turn_left()

    robot.move_forward()
    robot.turn_left()

    robot.move_forward()
    robot.turn_left()
