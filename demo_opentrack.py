# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
import cv2
import numpy as np
import yaml
from collections import deque
import socket
import time
import math
RAD2DEG = 180./math.pi

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.functions import cv_draw_landmark
from utils.pose import viz_pose, calc_pose, P2sRt

def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    gpu_mode = args.mode == 'gpu'
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    vis = args.vis
    UDP_IP = args.ip
    UDP_PORT = args.port
    fov = args.fov

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, \
        imageio.get_reader("<video0>") as reader:

        # Initialize FaceBoxes
        face_boxes = FaceBoxes()

        dense_flag = False
        pre_ver = None
        first = True
        for frame in reader:
            frame_bgr = frame[..., ::-1]  # RGB->BGR
            #inference_time = time.time()
            if first:
                # the first frame, detect face, here we only use the first face, you can change depending on your need
                boxes = face_boxes(frame_bgr)
                if boxes:
                    boxes = [boxes[0]]
                    param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
                    ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
                    # refine
                    param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
                    ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
                    first = False
            else:
                param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

                roi_box = roi_box_lst[0]
                # todo: add confidence threshold to judge the tracking is failed
                if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                    boxes = face_boxes(frame_bgr)
                    if boxes:
                        boxes = [boxes[0]]
                        param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
                    else:
                        first = True
                        pre_ver = None
                if boxes:
                    ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            #inference_time = time.time()-inference_time
            if boxes:
                pre_ver = ver  # for tracking
                # Most of what follows here is an attempt to compute the translation t
                # of the head model in world space. The depth coordinate shall be denoted tz.
                h, w, _ = frame.shape
                f, R, t = P2sRt(param_lst[0][:12].reshape(3, -1))
                P, pose = calc_pose(param_lst[0])
                x0, y0, x1, y1 = map(int, roi_box_lst[0])
                # Compute the translation vector in image space.
                # The prediction of t is defined in some arbitrary scaled way (from our perspective here.)
                # See tddfa_util.similar_transform(), which computes the orthographic 
                # projection of 3d mesh positions to the image space of our roi_box.
                t_img_x = x0 + (t[0]-1) * (x1-x0) / tddfa.size
                t_img_y = y0 + (tddfa.size-t[1]) * (y1-y0) / tddfa.size
                # The following should give us the scaling transfrom from the 
                # space of the (deformed) model to image pixels.
                scale_model_to_roi_x = f / tddfa.size * (x1-x0)
                # Probably the y "version" makes no sense ... Idk ...
                scale_model_to_roi_y = f / tddfa.size * (y1-y0)
                # In order to relate length in our roi box to the real world we can utilize
                # the size of the deformable model. Judging from magnitude of the  coordinate values 
                # in the model files (https://github.com/cleardusk/3DDFA/tree/master/, u_exp.npy and u_shp.npy),
                # they are defined in micrometers. Hence ...
                scale_model_to_meters = 1.e-6   # micrometers to meters.
                # Furthermore let's define the focal length of our webcam considered as pinhole camera. 
                # (I think this is focal length, or rather one over it)
                focal_length_inv_x = math.tan(fov*math.pi/180.*0.5)
                focal_length_inv_y = math.tan(fov*h/w*math.pi/180.*0.5)
                # Considering the perspective transform, we know that tz/world_space_size = f/projected_size,
                # where projected_size ranges from -1 to 1 as it is defined in screen space. Or rearanging slightly.
                # tz = focal_length * projected_size/world_space_size. The quotient in the back is just a scaling
                # factor which we can compute with the quantities we have. I.e. scale_model_to_roi_x/w gives us
                # the model->screen scale trafo. The inverse screen->model. Multiply by model->meters, and we have
                # our desired meters->screen scaling.
                # Uhm ... except I have this twice now, one for x, one for y ...
                tz_roi_x = scale_model_to_meters * w / (scale_model_to_roi_x * focal_length_inv_x)
                tz_roi_y = scale_model_to_meters * h / (scale_model_to_roi_y * focal_length_inv_y)
                # Just average ... 
                tz_roi = (tz_roi_x + tz_roi_y)*0.5
                # I don't know how the length units of the predicted translation depth are defined.
                # This scaling factor does not seem to bad though.
                scale_model_z = scale_model_to_meters / f
                # So we just add the scaled predicted depth to the guestimated depth from the head size.
                tz = tz_roi + t[2]*scale_model_z
                # Using the perspective transform, calculating the world x and y translation is easy.
                tx = (t_img_x / w * 2. - 1.) * tz * focal_length_inv_x
                ty = (t_img_y / h * 2. - 1.) * tz * focal_length_inv_y
                
                # This final part corrects the yaw and pitch values. Why is simple. The network only sees the
                # roi around the face. When the roi moves off-center our head is seen at an angle due to perspective
                # even though in world space we face straight forward in z-direction. By adding the angle between
                # the forward direction and the head translation vector, we can compensate for this effect.
                yaw_correction = math.atan2(tx, tz) * RAD2DEG
                pitch_correction = math.atan2(ty, tz) * RAD2DEG
                pose[0] += yaw_correction
                pose[1] += pitch_correction

                print (f"H {pose[0]:.1f} P {pose[1]:.1f} B {pose[2]:.1f}" + 
                       f"X {tx:.2f} Y {ty:.2f} Z {tz:.2f}")
                
                # Send the pose to opentrack
                a = np.zeros((6,), dtype=np.float64)
                # I guess it wants cm ...
                a[0] = tx * 100.
                a[1] = ty * 100.
                a[2] = tz * 100.
                # Degrees are fine.
                a[3] = pose[0]
                a[4] = pose[1]
                a[5] = pose[2]
                sock.sendto(a.tobytes(), (UDP_IP, UDP_PORT))
                #print ("update period ", time.time()-time_val, "inference time", inference_time)
                #time_val = time.time()

                if vis:
                    img_draw = cv_draw_landmark(frame_bgr, ver)  # since we use padding
                    viz_pose(img_draw, param_lst, [ver])
                    # The 2d bounding box
                    cv2.rectangle(img_draw, (x0, y0), (x1, y1) ,(128,0,0),1)
                    # Draw the projection of a 10cm radius circle. Sanity check for the 
                    # scaling and the translation estimation.
                    cx = int((tx / (focal_length_inv_x*tz) + 1.)*0.5*w)
                    cy = int((ty / (focal_length_inv_y*tz) + 1.)*0.5*h)
                    r = int(0.1 * w / (focal_length_inv_x*tz))
                    cv2.circle(img_draw, (cx, cy), r, (0,0,255), 1)
                    cv2.circle(img_draw, (cx, cy), 3, (0,0,255), -1)
                    cv2.imshow('image', img_draw)
                    cv2.waitKey(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose transmission to Opentrack')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-f', '--fov', default=60., type=float, help='field of view angle of your camera')
    parser.add_argument('-p', '--port', default=4242, type=int, help='port to communicate with opentrack')
    parser.add_argument('-i', '--ip', default="127.0.0.1", type=str, help='ip to communicate with opentrack')
    parser.add_argument('-v', '--vis', default=False, type=bool, help='enable visualization')
    args = parser.parse_args()
    main(args)
