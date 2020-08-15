import numpy as np
import os
import cv2
import shutil
from tqdm import tqdm
import sys
from static_transforms import *
from scipy.spatial.transform import Rotation as R


def get_camera_timestamps(data_folder_path):
    timestamp_filepaths_dict = {
        "forward_camera":
        os.path.join(data_folder_path, "forward_camera/timestamps.txt"),
        "left_camera":
        os.path.join(data_folder_path, "left_camera/timestamps.txt"),
        "right_camera":
        os.path.join(data_folder_path, "right_camera/timestamps.txt")
    }

    return read_timestamps(timestamp_filepaths_dict)


def get_lidar_timestamps(data_folder_path):
    timestamp_filepaths_dict = {
        "fw_lidar": os.path.join(data_folder_path, "fw_lidar/timestamps.txt"),
        "mrh_lidar": os.path.join(data_folder_path, "mrh_lidar/timestamps.txt")
    }

    return read_timestamps(timestamp_filepaths_dict)


def get_image_size(data_folder_path):
    img = cv2.imread(
        os.path.join(data_folder_path, "forward_camera/00000000.png"))
    return img.shape


def read_timestamps(timestamp_filepaths_dict):
    timestamp_arrays_dict = {}
    for key in timestamp_filepaths_dict:
        timestamp_arrays_dict[key] = []

    for key in timestamp_filepaths_dict:
        with open(timestamp_filepaths_dict[key]) as timestamps_file:
            for timestamp in timestamps_file:
                timestamp_arrays_dict[key].append(timestamp)
            timestamp_arrays_dict[key] = np.array(timestamp_arrays_dict[key],
                                                  dtype=np.float)
    return timestamp_arrays_dict


def timestamps_within_interval(interval, timestamps):
    min_timestamp = np.min(np.array(timestamps))
    max_timestamp = np.max(np.array(timestamps))
    return max_timestamp - min_timestamp < interval


def write_reference_timestamps(dst_image_folder, reference_timestamps):
    print("Update timestamps.txt")
    with open(os.path.join(dst_image_folder, 'timestamps.txt'),
              'w') as filehandle:
        filehandle.writelines("{:.6f}\n".format(timestamp)
                              for timestamp in reference_timestamps)


def lists_in_dict_empty(dict_of_lists):
    total_num_elements = 0
    for key in dict_of_lists:
        total_num_elements += len(dict_of_lists[key])

    if total_num_elements != 0:
        return False
    else:
        return True


def read_point_cloud(point_cloud_file):
    _, ext = os.path.splitext(point_cloud_file)

    if ext == '.bin':
        point_cloud = (np.fromfile(point_cloud_file,
                                   dtype=np.float32).reshape(-1, 6))
    elif ext == '.npy':
        point_cloud = (np.load(point_cloud_file,
                               dtype=np.float32).reshape(-1, 6))
    else:
        print("Invalid point cloud format encountered.")
        sys.exit()

    return point_cloud


def read_static_transformation(transform):
    if transform == "fw_lidar_to_mrh_lidar":
        yaw = fw_lidar_to_mrh_lidar[20200726111500]['yaw']
        pitch = fw_lidar_to_mrh_lidar[20200726111500]['pitch']
        roll = fw_lidar_to_mrh_lidar[20200726111500]['roll']
        x = fw_lidar_to_mrh_lidar[20200726111500]['x']
        y = fw_lidar_to_mrh_lidar[20200726111500]['y']
        z = fw_lidar_to_mrh_lidar[20200726111500]['z']
        r = R.from_euler('zyx', [yaw, pitch, roll], degrees=False)
        t = np.array([x, y, z])

    elif transform == "mrh_lidar_to_egomotion":
        yaw = mrh_lidar_to_egomotion[20200726111500]['yaw']
        pitch = mrh_lidar_to_egomotion[20200726111500]['pitch']
        roll = mrh_lidar_to_egomotion[20200726111500]['roll']
        x = mrh_lidar_to_egomotion[20200726111500]['x']
        y = mrh_lidar_to_egomotion[20200726111500]['y']
        z = mrh_lidar_to_egomotion[20200726111500]['z']
        r = R.from_euler('zyx', [yaw, pitch, roll], degrees=False)
        t = np.array([x, y, z])

    else:
        print("The requested static transform doesn't exist")

    T = np.zeros((4, 4))
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = t
    T[3, 3] = 1

    return T


def read_dynamic_transformation(transform, data_folder_path):
    if transform == "egomotion_to_world":
        file_path = os.path.join(data_folder_path, "tf/egomotion_to_world.txt")

        return np.loadtxt(file_path, delimiter=",")

    else:
        print("The requested dynamic transform doesn't exist")