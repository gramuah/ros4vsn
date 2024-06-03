#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


class ImagePreprocessing:
    """
    This class is in charge of reading image topic, and convert it to a ROS message from a numpy array
    """
    def __init__(self, sub_rgb_name: str = "/camera/color/image_raw", sub_depth_name: str = "/camera/depth/image_raw", median_filter: int = 10) -> None:
        self.bridge = CvBridge()
        self._sub_rgb_name: str = sub_rgb_name
        self._sub_depth_name: str = sub_depth_name
        self._rgb_image = None
        self._depth_image = None
        self._array_images = []
        self._contador_array = 0
        self.median_filter = median_filter

        rospy.loginfo("Connecting camera...")
        # ROS subscriber to the camera topic, if its another one we will need to change its value
        rospy.Subscriber(self._sub_rgb_name, Image, self._callback_rgb)
        rospy.Subscriber(self._sub_depth_name, Image, self._callback_depth )

    def _callback_rgb(self, data: Image) -> np.array:
        try:
            self._rgb_image = self.bridge.imgmsg_to_cv2(data, "rgb8")

        except CvBridgeError as e:
            print(e)

    def get_image_rgb(self):
        return self._rgb_image

    def clean_array(self):
        self._array_images = []
    def _callback_depth(self, data: Image) -> np.array:
        try:
            a = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

            imagen = a/10000

            self._array_images.append(imagen)

        except CvBridgeError as e:
            print(e)
    def array_ready(self) -> bool:
        if len(self._array_images) < self.median_filter:
            return False
        else:
            return True



    def get_image_depth(self):
        median = np.median(self._array_images, axis=0)
        self._depth_image = median
        return self._depth_image