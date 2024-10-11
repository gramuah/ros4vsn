import rospy

from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

from math import degrees
import numpy as np


class Odom:
    """
    This class is in charge of reading image topic, and convert it to a ROS message from a numpy array
    """
    def __init__(self, sub_name: str = "/odom") -> None:
        self._sub_name: str = sub_name
        self._actual_angle = None
        self._actual_position = None

        # ROS subscriber to the camera topic, if its another one we will need to change its value
        rospy.Subscriber(self._sub_name, Odometry, self._callback_odom)

    def _callback_odom(self, data: Odometry):
        self._actual_position = np.array((data.pose.pose.position.x, data.pose.pose.position.y, 0))
        q1 = data.pose.pose.orientation.x
        q2 = data.pose.pose.orientation.y
        q3 = data.pose.pose.orientation.z
        q4 = data.pose.pose.orientation.w

        e = euler_from_quaternion((q1, q2, q3, q4))
        th = degrees(e[2])
        self._actual_angle = e[2]

    def get_actual_position(self):
        return self._actual_position[0], self._actual_position[1], self._actual_angle