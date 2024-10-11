#!/usr/bin/env python3

import sys
from math import floor
import numpy as np
import time
import os
from datetime import datetime
import csv
import signal
import cv2
import matplotlib.pyplot as plt
import os
import rospy
show_screen_images = rospy.get_param('/vsn/show_screen_images')
new_row = []
pirlnav_path = rospy.get_param('/vsn/pirlnav_path')
archive_path = os.path.abspath(__file__)
relative_path = os.path.dirname(archive_path)
log_route = relative_path + '/log'
sys.path.append(pirlnav_path)
from discrete_move.srv import DiscreteServer, DiscreteServerResponse
from odometry import Odom
from class_model import Pirlnav
from image_preprocessing import ImagePreprocessing
from types import SimpleNamespace
from torchvision import transforms, models
import torch
import torch.nn.functional as F

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src.prepare_data import prepare_data

from PIL import Image

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
    semantic_directory = base_directory + '/semantic'
    depth_directory = base_directory + '/depth'
    print(rgb_directory)
    try:
        os.makedirs(rgb_directory, exist_ok=True)
        os.makedirs(depth_directory, exist_ok=True)
        os.makedirs(semantic_directory, exist_ok=True)
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


    # dataset
    segmentor_path = '/home/gram/rafarepos/ros4vsn/examples/semnav/oursegmodel.pth'

    #Define the rgb image transform
    transform = transforms.Compose([
        transforms.Resize((480, 640)),  # Ajustar al tamaño usado en el entrenamiento
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #As were using 40 classes, our constant must be this to use all colors
    color_constant = np.floor((255*255*255)/40)
    #Load our image segmentation model
    seg_model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    seg_model.classifier[4] = torch.nn.Conv2d(256, 41, kernel_size=(1, 1))  # Ajustar a 41 clases si es necesario
    seg_model.load_state_dict(torch.load(segmentor_path), strict=False)
    seg_model.eval()
    ###
    num_action = 0
    max_action = 1500000
    time.sleep(1)

    while 1:
        time.sleep(0.5)
        image = preprocess.get_image_rgb()

        if image is not None:
            ##Generate semantic photo using rgb image
            input_image = transform(Image.fromarray(image, 'RGB')).unsqueeze(0)
            with torch.no_grad():  # No calcular gradientes
                output = seg_model(input_image)['out']  # Obtener la salida del modelo
                output_predictions = output.argmax(dim=1)  # Obtener la clase con mayor probabilidad

            # Transform our semantic prediction into rgb semantic prediction
            output_predictions = output_predictions.squeeze(0).cpu().numpy()
            output_predictions = output_predictions*color_constant
            rgb_matrix = np.zeros((480, 640, 3), dtype=np.uint32)
            rgb_matrix[:, :, 0] = (output_predictions.astype(np.uint64) >> 16) & 0xFF  # R
            rgb_matrix[:, :, 1] = (output_predictions.astype(np.uint64) >> 8) & 0xFF  # G
            rgb_matrix[:, :, 2] = output_predictions.astype(np.uint64) & 0xFF  # B



            #Save both, rgb and semantic rgb photos
            cv2.imwrite(semantic_directory + "/" + str(num_action) + ".png", cv2.cvtColor(rgb_matrix.astype(np.uint8),cv2.COLOR_RGB2BGR))
            cv2.imwrite(rgb_directory + "/" + str(num_action) + ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            if show_screen_images:
                cv2.imshow("  image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break

            # Add new action
            num_action = num_action + 1

            position = odometry.get_actual_position()

            # IA class
            input_observation = [{'rgb': image,
                                 'objectgoal': object_goal,
                                  'semantic': output_predictions,
                                  'semantic_rgb': rgb_matrix,
                                  'compass': position[2],
                                 'gps': np.array([position[0], position[1]])}]
            action = model.evaluation(observation=input_observation)
            actions.append(action)

            new_row = [tim, object_goal, actions, num_action]

            check_depth()

            if action == 1:

                if check_depth():
                    print("Possible colission")
                    continue

            print(f"Steps {num_action} - Action: {action_dict[action]}")

            server_response = send_action_service(action_dict[action], 30)
            # time.sleep(1)

            if not server_response:
                rospy.loginfo("An error has been detected from the server")
                break

            if action == 0:
                rospy.loginfo("Robot reached the goal satisfactorily")
                server_response = send_action_service("Left", 90)

    sys.exit()