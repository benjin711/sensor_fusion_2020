import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

from utils.utils import *

# Pathing
forward_calib_path = '/home/carter/git/sensor_fusion_2020/preprocessing/preprocessing/resources/forward.yaml'

cone_ext = '.bin'
img_ext = '.png'
base_dir = '/media/carter/Samsung_T5/sensor-fusion-data/2020-07-05_tuggen/data/short2'
cone_dir = os.path.join(base_dir, 'cones_filtered')
forward_dir = os.path.join(base_dir, 'forward_camera_filtered')
index = 2

# Load Data
forward_calib = load_camera_calib(forward_calib_path)

cone_path = os.path.join(cone_dir, str(index).zfill(8) + cone_ext)
cone_data = np.fromfile(cone_path).reshape((-1, 4))
cone_xyz = cone_data[:, 1:]

forward_img_path = os.path.join(forward_dir, str(index).zfill(8) + img_ext)
forward_img = cv.imread(forward_img_path)

# Visualization Checks
# cv.imshow(f"Do you see {cone_xyz.shape[0]} cones?", forward_img)
# cv.waitKey(0)

plt.scatter(cone_xyz[:, 0], cone_xyz[:, 1])
plt.show()

# Transform from egomotion to camera frame
R_egomotion2forwardcamera = R.from_euler("zx", [-90, 90], degrees=True).as_matrix()
x_trans, y_trans, z_trans = 0., 0., 0.
x_rot, y_rot, z_rot = 0., 0., 0.

R_correction = R.from_euler("zyx", [-2.5, -1.0, -1], degrees=True).as_matrix()
R_total = np.matmul(R_correction, R_egomotion2forwardcamera)

def print_extrinsics():
    print(f"R_x = {x_rot}")
    print(f"R_y = {x_rot}")
    print(f"R_z = {x_rot}")
    print(f"T_x = {x_trans}")
    print(f"T_y = {x_trans}")
    print(f"T_z = {x_trans}")

# Projection
def draw_points(img, points, K):
    """
    Given an image and a set of points, draw each point using the projection matrix.
    :param img: (R, C) image
    :param points: (N, 3) array of 3D points
    """
    temp_img = img.copy()
    img_points = np.matmul(K, np.transpose(points))
    img_points /= img_points[2, :]
    img_points = np.transpose(img_points)

    in_bounds = 0
    for point_index in range(points.shape[0]):
        img_point = img_points[point_index, :]

        if 0 <= img_point[0] <= img.shape[1] and 0 <= img_point[1] <= img.shape[1]:
            in_bounds += 1
            img = cv.circle(temp_img, (int(img_point[0]), int(img_point[1])), 2, (255, 0, 0), -1)

    cv.imshow("Projection", temp_img)
    cv.waitKey(0)

x_lims = [-1.0, 1.0]
y_lims = [-1.0, 1.0]
z_lims = [-1.0, 1.0]
trans_resolution = 100

def x_trackbar(x_bar_val):
    global x_trans

    x_trans_new = (x_bar_val/trans_resolution)*(x_lims[1]-x_lims[0])+x_lims[0]
    x_trans = x_trans_new

    T = np.array([x_trans, y_trans, z_trans]).reshape((3, 1))
    cone_xyz_new = np.transpose(np.matmul(R_total, np.transpose(cone_xyz)))
    cone_xyz_new += np.transpose(T)
    draw_points(forward_img, cone_xyz_new, forward_calib['camera_matrix'])
    print_extrinsics()

def y_trackbar(y_bar_val):
    global y_trans

    y_trans_new = (y_bar_val / trans_resolution) * (y_lims[1] - y_lims[0]) + \
                  y_lims[0]
    y_trans = y_trans_new

    T = np.array([x_trans, y_trans, z_trans]).reshape((3, 1))
    cone_xyz_new = np.transpose(np.matmul(R_total, np.transpose(cone_xyz)))
    cone_xyz_new += np.transpose(T)
    draw_points(forward_img, cone_xyz_new, forward_calib['camera_matrix'])
    print_extrinsics()

def z_trackbar(z_bar_val):
    global z_trans

    z_trans_new = (z_bar_val / trans_resolution) * (z_lims[1] - z_lims[0]) + \
                  z_lims[0]
    z_trans = z_trans_new

    T = np.array([x_trans, y_trans, z_trans]).reshape((3, 1))
    cone_xyz_new = np.transpose(np.matmul(R_total, np.transpose(cone_xyz)))
    cone_xyz_new += np.transpose(T)
    draw_points(forward_img, cone_xyz_new, forward_calib['camera_matrix'])
    print_extrinsics()

x_rot_lims = [-10., 10.]
y_rot_lims = [-10., 10.]
z_rot_lims = [-10., 10.]
rot_resolution = 200

def x_rot_trackbar(x_bar_val):
    global x_rot

    x_rot_new = (x_bar_val/trans_resolution)*(x_rot_lims[1]-x_rot_lims[0])+x_rot_lims[0]
    x_rot = x_rot_new

    R_correction = R.from_euler("zyx", [z_rot, y_rot, x_rot],
                                degrees=True).as_matrix()
    R_total = np.matmul(R_correction, R_egomotion2forwardcamera)
    T = np.array([x_trans, y_trans, z_trans]).reshape((3, 1))

    cone_xyz_new = np.transpose(np.matmul(R_total, np.transpose(cone_xyz)))
    cone_xyz_new += np.transpose(T)
    draw_points(forward_img, cone_xyz_new, forward_calib['camera_matrix'])
    print_extrinsics()

def y_rot_trackbar(y_bar_val):
    global y_rot

    y_rot_new = (y_bar_val / trans_resolution) * (y_rot_lims[1] - y_rot_lims[0]) + \
                y_rot_lims[0]
    y_rot = y_rot_new

    R_correction = R.from_euler("zyx", [z_rot, y_rot, x_rot],
                                degrees=True).as_matrix()
    R_total = np.matmul(R_correction, R_egomotion2forwardcamera)
    T = np.array([x_trans, y_trans, z_trans]).reshape((3, 1))

    cone_xyz_new = np.transpose(np.matmul(R_total, np.transpose(cone_xyz)))
    cone_xyz_new += np.transpose(T)
    draw_points(forward_img, cone_xyz_new, forward_calib['camera_matrix'])
    print_extrinsics()

def z_rot_trackbar(z_bar_val):
    global z_rot

    z_rot_new = (z_bar_val / trans_resolution) * (z_rot_lims[1] - z_rot_lims[0]) + \
                z_rot_lims[0]
    z_rot = z_rot_new

    R_correction = R.from_euler("zyx", [z_rot, y_rot, x_rot],
                                degrees=True).as_matrix()
    R_total = np.matmul(R_correction, R_egomotion2forwardcamera)
    T = np.array([x_trans, y_trans, z_trans]).reshape((3, 1))

    cone_xyz_new = np.transpose(np.matmul(R_total, np.transpose(cone_xyz)))
    cone_xyz_new += np.transpose(T)
    draw_points(forward_img, cone_xyz_new, forward_calib['camera_matrix'])
    print_extrinsics()

# draw_points(forward_img, cone_xyz, forward_calib['camera_matrix'])
cv.namedWindow("Projection")
cv.imshow("Projection", forward_img)
cv.createTrackbar('X Translation', "Projection", 0, trans_resolution, x_trackbar)
cv.createTrackbar('Y Translation', "Projection", 0, trans_resolution, y_trackbar)
cv.createTrackbar('Z Translation', "Projection", 0, trans_resolution, z_trackbar)
cv.createTrackbar('X Rotation', "Projection", 0, rot_resolution, x_rot_trackbar)
cv.createTrackbar('Y Rotation', "Projection", 0, rot_resolution, y_rot_trackbar)
cv.createTrackbar('Z Rotation', "Projection", 0, rot_resolution, z_rot_trackbar)
cv.waitKey()
