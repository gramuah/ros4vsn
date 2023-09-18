#!/usr/bin/env python3

import sys
import rospy
from discrete_move.srv import DiscreteServer, DiscreteServerResponse

from image_preprocessing import ImagePreprocessing
from ia import Ia


def send_action_service(mov: str, ang: int, service_name: str = "discrete_move") -> bool:
    rospy.wait_for_service(service_name, timeout=10.0)

    try:
        serv = rospy.ServiceProxy(service_name, DiscreteServer)
        resp = serv(mov, ang)
        return resp.response

    except (rospy.ServiceException, rospy.ROSException):
        return False


if __name__ == "__main__":
    rospy.init_node('vsn_node', anonymous=True)

    # Load configure svn
    object_goal = rospy.get_param('/vsn/goal')
    sub_name = rospy.get_param('/vsn/camera_topic')

    preprocess = ImagePreprocessing(sub_name)
    ia = Ia()

    rospy.loginfo(f"I'm searching a {object_goal}")

    while 1:
        image = preprocess.get_image_rgb()
        if image is not None:

            # IA class
            action, angle = ia.random()

            server_response = send_action_service(action, angle)

            if not server_response:
                rospy.loginfo("An error has been detected from the server")
                break

            if action == "Stop":
                rospy.loginfo("Robot reached the goal satisfactorily")
                break

    sys.exit()
