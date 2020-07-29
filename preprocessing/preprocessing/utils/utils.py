import numpy as np
import os
import cv2
import shutil
from tqdm import tqdm


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
