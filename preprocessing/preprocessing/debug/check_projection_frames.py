import pkg_resources
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rotation
from scipy.spatial.ckdtree import cKDTree
import scipy.signal as signal
import copy
import os
import csv

from utils.utils import *

# Pathing
forward_calib_path = '/home/carter/git/sensor_fusion_2020/preprocessing/preprocessing/resources/forward.yaml'

cone_ext = '.bin'
img_ext = '.png'
base_dir = '/media/carter/Samsung_T5/sensor_fusion_data/2020-07-12_tuggen/data/autocross_2020-07-12-10-00-35'
cone_dir = os.path.join(base_dir, 'cones_filtered')
forward_dir = os.path.join(base_dir, 'forward_camera_filtered')
gnss_path = os.path.join(base_dir, 'gnss_filtered', 'gnss.bin')

# Load Data
gnss_data = np.fromfile(gnss_path).reshape((-1, 7))
forward_calib = load_camera_calib(forward_calib_path)
cone_offset = 0


def draw_points(image, points, K, R=np.eye(3),
                T=np.array([0, 0, 0]).reshape((3, 1)), window_name="Blank",
                pause=False):
    points_xform = np.transpose(np.matmul(R, np.transpose(points)))
    points_xform += np.transpose(T)
    pixels = np.transpose(np.matmul(K, np.transpose(points_xform)))
    pixels /= pixels[:, 2].reshape((-1, 1))
    pixels = pixels.astype(np.int)

    new_image = image.copy()
    for pixel_idx in range(pixels.shape[0]):
        cv.circle(new_image, (pixels[pixel_idx, 0], pixels[pixel_idx, 1]),
                  radius=2, color=(255, 0, 255), thickness=-1)

        if pause:
            color_code = cone_color[pixel_idx]
            if color_code == 0:
                print('Blue')
            elif color_code == 1:
                print('Yellow')
            print(pixel_idx)
            cv.imshow(window_name, new_image)
            cv.waitKey(0)
            cv.destroyAllWindows()
    print('Rotation Matrix')
    print(R)
    print('Translation Vector')
    print(T)
    cv.imshow(window_name, new_image)
    cv.waitKey(0)
    return new_image

for index in range(0, 250, 1):
    cone_path = os.path.join(cone_dir, str(index).zfill(8) + cone_ext)
    cone_data = np.fromfile(cone_path).reshape((-1, 4))
    cone_xyz = cone_data[:, 1:]
    cone_color = cone_data[:, 0]

    forward_img_path = os.path.join(forward_dir, str(index).zfill(8) + img_ext)
    forward_img = cv.imread(forward_img_path)
    forward_img = cv.undistort(forward_img, forward_calib['camera_matrix'],
                               forward_calib['distortion_coefficients'])
    first_proj = draw_points(forward_img, cone_xyz, forward_calib['camera_matrix'], window_name="First Proj")


