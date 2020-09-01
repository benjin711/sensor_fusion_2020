import numpy as np
import cv2 as cv
import os
from utils.utils import load_camera_calib, load_stereo_calib

calib_left = load_stereo_calib('../resources/left.yaml', '../resources/left_forward.yaml')
calib_right = load_stereo_calib('../resources/right.yaml', '../resources/right_forward.yaml')
calib_forward = load_camera_calib('../resources/forward.yaml')

# Compute L -> R transforms
R_fr = np.transpose(calib_right['rotation_matrix'])
T_fr = -np.matmul(R_fr, calib_right['translation_vector'])

R_lr = np.matmul(R_fr, calib_left['rotation_matrix'])
T_lr = np.matmul(R_fr, calib_left['translation_vector']) + T_fr


def cylindrical_projection(src, f, xc, yc):

    # Generate maps
    map_x, map_y = np.zeros(src.shape[:2]), np.zeros(src.shape[:2])
    for row in range(src.shape[0]):
        for col in range(src.shape[1]):
            map_x[row, col] = f*np.tan((col - xc)/f) + xc
            map_y[row, col] = (row - yc)/np.cos((col - xc)/f) + yc

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    return cv.remap(src, map_x, map_y, cv.INTER_LINEAR)

base_dir = '/media/carter/Samsung_T5/amz/bags/turbinenplatz-2/data/sensors_2020-06-16-13-01-18/'
left_dir = os.path.join(base_dir, 'left_camera')
forward_dir = os.path.join(base_dir, 'forward_camera')
right_dir = os.path.join(base_dir, 'right_camera')

test_image = cv.imread(os.path.join(left_dir, str(0).zfill(8) + '.png'))
test_mtx = calib_left['camera_matrix']
new_image = cylindrical_projection(test_image, test_mtx[0, 0], test_mtx[0, 2], test_mtx[1, 2])
cv.imshow('Original', test_image)
cv.imshow('Remapped', new_image)
cv.waitKey(0)