#!/usr/bin/env python3

import sys
import numpy as np
import time
import os
from datetime import datetime
import csv
import signal
import cv2

import os
import rospy
from discrete_move.srv import DiscreteServer, DiscreteServerResponse
from odometry import Odom
from class_model import Pirlnav
from image_preprocessing import ImagePreprocessing

show_screen_images = rospy.get_param('/vsn/show_screen_images')
new_row = []
pirlnav_path = rospy.get_param('/vsn/pirlnav_path')
archive_path = os.path.abspath(__file__)
relative_path = os.path.dirname(archive_path)
log_route = relative_path + '/log'
sys.path.append(pirlnav_path)


def handler(signum, frame):
    ruta_archivo = log_route + '/datos.csv'
    print(new_row)
    with open(ruta_archivo, 'a', newline='') as archivo_csv:
        escritor = csv.writer(archivo_csv)

        escritor.writerow(new_row)
    exit(1)


def check_depth():
    time.sleep(0.5)
    depth = preprocess.get_image_depth()
    print(depth)

    threshold = 0.1

    num_val = np.count_nonzero(depth < threshold)
    print(num_val)

    if num_val > 150000:
        return True

    return False


def send_action_service(mov: str, ang: int, service_name: str = "discrete_move") -> bool:
    rospy.wait_for_service(service_name, timeout=10.0)

    try:
        serv = rospy.ServiceProxy(service_name, DiscreteServer)
        resp = serv(mov, ang)
        return resp.response

    except (rospy.ServiceException, rospy.ROSException):
        return False


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handler)

    now = datetime.now()
    actions = []
    tim = now.strftime("%d_%m_%H_%M_%S")

    base_directory = log_route + tim
    rgb_directory = base_directory + '/rgb'
    depth_directory = base_directory + '/depth'

    try:
        os.makedirs(rgb_directory, exist_ok=True)
        os.makedirs(depth_directory, exist_ok=True)
        print(f'Directory  was created successfully')
    except OSError as error:
        print(error)

    rospy.init_node('vsn_node', anonymous=True)

    action_dict = {0: "Stop",
                   1: "Forward",
                   2: "Left",
                   3: "Right",
                   4: "Backward",
                   5: "Forward", }

    # Load configure svn
    object_goal = rospy.get_param('/vsn/goal')
    sub_name = rospy.get_param('/vsn/camera_topic')

    preprocess = ImagePreprocessing(sub_name)
    model = Pirlnav()
    odometry = Odom()

    rospy.loginfo(f"I'm searching a {object_goal}")

    num_action = 0
    max_action = 1500000
    time.sleep(1)

    while 1:
        time.sleep(0.5)
        image = preprocess.get_image_rgb()

        if image is not None:
            cv2.imwrite(rgb_directory + "/" + str(num_action) + ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if show_screen_images:
                cv2.imshow("  image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break

            # Add new action
            num_action = num_action + 1

            position = odometry.get_actual_position()

            # IA class
            input_observation = {'image': image,
                                 'obj_goal': object_goal,
                                 'position': np.array([position[0], position[1], position[2]])}

            action = model.evaluation(observation=input_observation)
            actions.append(action)

            new_row = [tim, object_goal, actions, num_action]

            check_depth()

            if action[0] == 1:

                if check_depth():
                    print("Posible ostion")
                    continue

            print(f"Steps {num_action} - Action: {action_dict[action[0]]}")

            server_response = send_action_service(action_dict[action[0]], 30)
            # time.sleep(1)

            if not server_response:
                rospy.loginfo("An error has been detected from the server")
                break

            if action[0] == 0:
                rospy.loginfo("Robot reached the goal satisfactorily")
                server_response = send_action_service("Left", 90)

    sys.exit()
