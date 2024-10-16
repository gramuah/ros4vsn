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
sys.path.append('/home/rafa/repositorios/ESANet')
from discrete_move.srv import DiscreteServer, DiscreteServerResponse
from odometry import Odom
from class_model import Pirlnav
from image_preprocessing import ImagePreprocessing
from types import SimpleNamespace

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
    segmentor_path = '/home/rafa/repositorios/ros4vsn/examples/semnav/seg_models/nyuv2/r34_NBt1D_scenenet.pth'
    args  = SimpleNamespace(activation='relu',aug_scale_max=1.4,aug_scale_min=1.0,batch_size=8,batch_size_valid=None,c_for_logarithmic_weighting=1.02, channels_decoder=128, ckpt_path=segmentor_path, class_weighting='median_frequency', context_module='ppm', dataset='nyuv2', dataset_dir=None, debug=False, decoder_channels_mode='decreasing', depth_scale=0.1, encoder='resnet34', encoder_block='NonBottleneck1D', encoder_decoder_fusion='add', encoder_depth=None, epochs=500, finetune=None, freeze=0, fuse_depth_in_rgb_encoder='SE-add', he_init=False, height=480, last_ckpt="", lr=0.01, modality='rgbd', momentum=0.9, nr_decoder_blocks=[3], optimizer='SGD', pretrained_dir='./trained_models/imagenet', pretrained_on_imagenet=False, pretrained_scenenet="", raw_depth=True, upsampling='learned-3x3-zeropad', valid_full_res=False, weight_decay=0.0001, width=640, workers=8)
    args.pretrained_on_imagenet = False  # we are loading other weights anyway
    dataset, preprocessor = prepare_data(args, with_input_orig=True)
    n_classes = dataset.n_classes_without_void
    color_constant = np.floor((255*255*255)/40)

    # model and checkpoint loading
    seg_model, device = build_model(args, n_classes=n_classes)
    checkpoint = torch.load(args.ckpt_path,
                            map_location=lambda storage, loc: storage)
    seg_model.load_state_dict(checkpoint['state_dict'])
    print('Loaded checkpoint from {}'.format(args.ckpt_path))

    seg_model.eval()
    seg_model.to(device)
    ###
    num_action = 0
    max_action = 1500000
    time.sleep(1)

    while 1:
        time.sleep(0.5)
        image = preprocess.get_image_rgb()

        if image is not None:

            ##Generate semantic photo
            depth_image = preprocess.get_image_depth()
            depth_image_uint32 = depth_image.astype(np.uint32)
            depth_image= depth_image.astype(np.float32)

            save_depth = Image.fromarray(depth_image_uint32)
            sample_to_semantic = preprocessor({'image': image, 'depth': depth_image})
            # add batch axis and copy to device
            image_sem = sample_to_semantic['image'][None].to(device)
            depth_sem = sample_to_semantic['depth'][None].to(device)

            # apply network
            pred = seg_model(image_sem, depth_sem)

            pred = F.interpolate(pred, (480, 640),
                                 mode='bilinear', align_corners=False)
            pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy().squeeze().astype(np.uint8)

            pred = pred + 1 #Add 1 because we have not void
            pred  = pred*color_constant

            rgb_matrix = np.zeros((480, 640, 3), dtype=np.int32)
            rgb_matrix[:, :, 0] = (pred.astype(np.uint64) >> 16) & 0xFF  # R
            rgb_matrix[:, :, 1] = (pred.astype(np.uint64) >> 8) & 0xFF  # G
            rgb_matrix[:, :, 2] = pred.astype(np.uint64) & 0xFF  # B
            ##Generate semantic photo


            cv2.imwrite(semantic_directory + "/" + str(num_action) + ".png", cv2.cvtColor(rgb_matrix.astype(np.uint8),cv2.COLOR_RGB2BGR))
            cv2.imwrite(rgb_directory + "/" + str(num_action) + ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(depth_directory +"/"+ str(num_action) + ".png", depth_image)
            save_depth.save(depth_directory +"/"+ str(num_action) + ".png")
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
                                  'semantic': pred,
                                  'semantic_rgb': rgb_matrix,
                                  'compass': position[2],
                                 'gps': np.array([position[0], position[1]])}]
            action = model.evaluation(observation=input_observation)
            actions.append(action)

            new_row = [tim, object_goal, actions, num_action]

            check_depth()

            if action == 1:

                if check_depth():
                    print("Posible ostion")
                    continue

            print(f"Steps {num_action} - Action: {action_dict[action]}")

            server_response = send_action_service(action_dict[action], 30)
            # time.sleep(1)

            if not server_response:
                rospy.loginfo("An error has been detected from the server")
                break

            if action == 0:
                rospy.loginfo("Robot reached the goal satisfactorily")
                #server_response = send_action_service("Left", 90)
                sys.exit()

    sys.exit()