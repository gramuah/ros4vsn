import rospy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from math import degrees
import math
import numpy as np

class Odom:
    """
    This class is in charge of reading odom topic, and return the robot position and compass
    """
    def __init__(self, sub_name: str = "/odom") -> None:
        self._sub_name: str = sub_name
        self._actual_angle = None
        self._actual_position = None
        self.q = None
        # ROS subscriber to the camera topic, if its another one we will need to change its value
        rospy.Subscriber(self._sub_name, Odometry, self._callback_odom)

    def to_positive_angle(self, th):

        while True:
            if th < 0:
                th += 360
            if th > 0:
                ans = th % 360
                return ans
                break

    def _callback_odom(self, data: Odometry):
        self._actual_position = np.array((data.pose.pose.position.x, data.pose.pose.position.y))
        q1 = data.pose.pose.orientation.x
        q2 = data.pose.pose.orientation.y
        q3 = data.pose.pose.orientation.z
        q4 = data.pose.pose.orientation.w
        self.q = (q1, q2, q3, q4)

    def get_actual_position(self):
        return np.asarray(self._actual_position)

    def get_actual_angle(self):
        if self.q is not None:
            e = euler_from_quaternion(self.q)
            th = degrees(e[2])
            full_angle = self.to_positive_angle(th)
            self._actual_angle = full_angle * math.pi / 180

        return self._actual_angle
