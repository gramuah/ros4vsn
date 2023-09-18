#!/usr/bin/env python3

import rospy
from ServerClass import Server


def main():
    rospy.init_node('server', log_level=rospy.INFO)
    Server()


if __name__ == '__main__':
    main()
