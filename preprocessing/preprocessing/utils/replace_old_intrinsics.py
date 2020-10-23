import yaml
import numpy as np
import os
from glob import glob
from shutil import copyfile

# F camera intrinsics
f_K_mtx = np.array([[1723.06, 0, 1301.8], [0, 1606.73, 128.883], [0, 0, 1]])
f_dist_coeff = np.array([-0.0121418, 0.0040215, -0.00237977, 0.000151017])

# L camera intrinsics
l_K_mtx = np.array([[1711.97, 0, 1324.42], [0, 1580.73, 114.478], [0, 0, 1]])
l_dist_coeff = np.array([0.0328083, -0.0216367, 0.00309066, 0.00124671])

# R camera intrinsics
r_K_mtx = np.array([[1724.2, 0, 1258.96], [0, 1580.73, 98.7964], [0, 0, 1]])
r_dist_coeff = np.array([0.0287106, 0.0149668, -0.0129965, -0.0145366])

base_folder_path = "/media/benjin/Samsung_T5/AMZ/sensor_fusion_data"

left_cam_intrinsic_paths = sorted(
    glob(
        os.path.join(base_folder_path, '*', 'static_transformations',
                     'left.yaml')))

for intrinsics_file in left_cam_intrinsic_paths:

    #copyfile(intrinsics_file, intrinsics_file + '.copy')

    with open(intrinsics_file, 'r') as f_read:
        try:
            camera_data = yaml.load(f_read, Loader=yaml.CLoader)
        except AttributeError:
            camera_data = yaml.load(f_read, Loader=yaml.Loader)

        camera_data['distortion_model'] = 'OPENCV'

        camera_data['camera_matrix']['data'] = f_K_mtx.reshape(9).tolist()

        camera_data['distortion_coefficients']['data'] = f_dist_coeff.tolist()
        camera_data['distortion_coefficients']['cols'] = 4

    with open(intrinsics_file, 'w') as f_write:
        yaml.dump(camera_data, f_write)
