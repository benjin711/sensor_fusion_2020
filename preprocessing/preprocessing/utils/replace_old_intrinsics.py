import yaml
import numpy as np
import os
from glob import glob
from shutil import copyfile

# Adjust this path and run script to update dataset intrinsics with latest intrinsics
BASE_FOLDER_PATH = "/media/benjin/Samsung_T5/AMZ/sensor_fusion_data"

intrinsics = {
    "forward": {
        "K_mtx":
        np.array([[1720.89, 0, 1305.88], [0, 1700.83, 123.925], [0, 0, 1]]),
        "dist_coeff":
        np.array([-0.165483, 0.0966005, 0.00094785, 0.00101802])
    },
    "left": {
        "K_mtx":
        np.array([[1709.13, 0, 1305.61], [0, 1704.14, 135.504], [0, 0, 1]]),
        "dist_coeff":
        np.array([-0.163319, 0.0934299, -0.000634316, -0.000454461])
    },
    "right": {
        "K_mtx":
        np.array([[1716.84, 0, 1308.49], [0, 1726.81, 118.542], [0, 0, 1]]),
        "dist_coeff":
        np.array([-0.164374, 0.0950635, -0.000300013, -0.000466711])
    }
}

for key in intrinsics.keys():

    cam_intrinsics_paths = sorted(
        glob(
            os.path.join(BASE_FOLDER_PATH, '*', 'static_transformations',
                         key + '.yaml')))

    for intrinsics_file in cam_intrinsics_paths:

        #copyfile(intrinsics_file, intrinsics_file + '.copy')

        with open(intrinsics_file, 'r') as f_read:
            try:
                camera_data = yaml.load(f_read, Loader=yaml.CLoader)
            except AttributeError:
                camera_data = yaml.load(f_read, Loader=yaml.Loader)

            camera_data['distortion_model'] = 'OPENCV'

            camera_data['camera_matrix']['data'] = intrinsics[key][
                "K_mtx"].reshape(9).tolist()

            camera_data['distortion_coefficients']['data'] = intrinsics[key][
                "dist_coeff"].tolist()
            camera_data['distortion_coefficients']['cols'] = 4

        with open(intrinsics_file, 'w') as f_write:
            yaml.dump(camera_data, f_write)
