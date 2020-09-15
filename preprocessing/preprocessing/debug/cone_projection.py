import pkg_resources
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rotation
import os
import csv

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
cone_color = cone_data[:, 0]

forward_img_path = os.path.join(forward_dir, str(index).zfill(8) + img_ext)
forward_img = cv.imread(forward_img_path)
forward_img = cv.undistort(forward_img, forward_calib['camera_matrix'],
                           forward_calib['distortion_coefficients'])

# Check Cones and Image separately
plt.figure()
plt.imshow(forward_img)
plt.figure()
plt.scatter(cone_xyz[:, 0], cone_xyz[:, 2])
plt.show()

def draw_points(image, points, K, R, T, window_name="Blank", pause=False):
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
    cv.imshow(window_name, new_image)
    cv.waitKey(0)


# # Initial Projection Check
# R = np.eye(3)
# T = np.array([0, 1.0, 0]).reshape((3, 1))
# draw_points(forward_img, cone_xyz, forward_calib['camera_matrix'], R, T)

# Correspondences
pixels = [[715, 330],
          [1027, 229],
          [1116, 197],
          [1672, 249],
          [1201, 171],
          [1542, 222],
          [1471, 206],
          [1283, 169],
          [1545, 164]]
pixels = np.asarray(pixels).astype(np.float64)

pixels2 = [[940, 261],
           [1082, 210],
           [1146, 189],
           [1354, 177],
           [1227, 170],
           [1539, 175],
           [1750, 163],
           [2409, 166],
           [2111, 185],
           [2496, 223]]
pixels2 = np.asarray(pixels2).astype(np.float64)

points_idxs2 = [0, 2, 4, 8, 15, 48, 22, 29, 64, 60]
points2 = cone_xyz[points_idxs2, :]
colors2 = cone_color[points_idxs2]

points_idxs = [0, 2, 4, 42, 14, 43, 44, 16, 20]
points = cone_xyz[points_idxs, :]
colors = cone_color[points_idxs]

p3p_idxs = [0, 3, 5, 9]
subset_idxs = [0, 3, 5, 9]
pixels = pixels[:, :]
points = points[:, :]

retval, rvec_p3p, tvec_p3p = cv2.solvePnP(points2[p3p_idxs, :], pixels2[p3p_idxs, :], forward_calib['camera_matrix'], forward_calib['distortion_coefficients'], flags=cv.SOLVEPNP_P3P)
# Check P3P Solution
rotmat_pnp, _ = cv.Rodrigues(rvec_p3p)
tvec_pnp = tvec_p3p
draw_points(forward_img, cone_xyz, forward_calib['camera_matrix'], rotmat_pnp, tvec_pnp, pause=False)

# Apply P3P Solution
cone_xyz = np.transpose(np.matmul(rotmat_pnp, np.transpose(cone_xyz)))
cone_xyz += np.transpose(tvec_pnp)

# Correction
dx, dy, dz = 0., 0., 0.
dxr, dyr, dzr = 0., 0., 0.
slider_max = 100
ranges = {'dxr': 5, 'dyr': 5, 'dzr': 5,
          'dx': 1.0, 'dy': 1.0, 'dz': 1.0}
import sys
sys.setrecursionlimit(15000)

def xr_trackbar(val):
    global dxr
    dxr = (val - slider_max/2)/slider_max*ranges['dxr']
    R = Rotation.from_euler('xyz', [dxr, dyr, dzr], degrees=True).as_matrix()
    T = np.array([dx, dy, dz]).reshape((3, 1))
    draw_points(forward_img, cone_xyz, forward_calib['camera_matrix'], R, T, "Window")

def yr_trackbar(val):
    global dyr
    dyr = (val - slider_max/2)/slider_max*ranges['dyr']
    R = Rotation.from_euler('xyz', [dxr, dyr, dzr], degrees=True).as_matrix()
    T = np.array([dx, dy, dz]).reshape((3, 1))
    draw_points(forward_img, cone_xyz, forward_calib['camera_matrix'], R, T, "Window")

def zr_trackbar(val):
    global dzr
    dzr = (val - slider_max/2)/slider_max*ranges['dzr']
    R = Rotation.from_euler('xyz', [dxr, dyr, dzr], degrees=True).as_matrix()
    T = np.array([dx, dy, dz]).reshape((3, 1))
    draw_points(forward_img, cone_xyz, forward_calib['camera_matrix'], R, T, "Window")

def x_trackbar(val):
    global dx
    dx = (val - slider_max / 2) / slider_max * ranges['dx']
    R = Rotation.from_euler('xyz', [dxr, dyr, dzr], degrees=True).as_matrix()
    T = np.array([dx, dy, dz]).reshape((3, 1))
    draw_points(forward_img, cone_xyz, forward_calib['camera_matrix'], R, T,
                "Window")

def y_trackbar(val):
    global dy
    dy = (val - slider_max / 2) / slider_max * ranges['dy']
    R = Rotation.from_euler('xyz', [dxr, dyr, dzr], degrees=True).as_matrix()
    T = np.array([dx, dy, dz]).reshape((3, 1))
    draw_points(forward_img, cone_xyz, forward_calib['camera_matrix'], R, T,
                "Window")

def z_trackbar(val):
    global dz
    dz = (val - slider_max / 2) / slider_max * ranges['dz']
    R = Rotation.from_euler('xyz', [dxr, dyr, dzr], degrees=True).as_matrix()
    T = np.array([dx, dy, dz]).reshape((3, 1))
    draw_points(forward_img, cone_xyz, forward_calib['camera_matrix'], R, T,
                "Window")

cv.imshow("Window", forward_img)
cv.createTrackbar("X Rotation", "Window" , slider_max//2, slider_max, xr_trackbar)
cv.createTrackbar("Y Rotation", "Window" , slider_max//2, slider_max, yr_trackbar)
cv.createTrackbar("Z Rotation", "Window" , slider_max//2, slider_max, zr_trackbar)
cv.createTrackbar("X Translation", "Window" , slider_max//2, slider_max, x_trackbar)
cv.createTrackbar("Y Translation", "Window" , slider_max//2, slider_max, y_trackbar)
cv.createTrackbar("Z Translation", "Window" , slider_max//2, slider_max, z_trackbar)
cv.waitKey()

