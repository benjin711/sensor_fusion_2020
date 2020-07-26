import numpy as np
import os
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


def filter_images(data_folder_path, idx_triples_dict, reference_timestamps,
                  keep_orig_image_folders):
    for key in idx_triples_dict:
        print("Filtering images in folder {}".format(key))
        src_image_folder_path = os.path.join(data_folder_path, key)
        dst_image_folder_path = os.path.join(data_folder_path,
                                             key + "_filtered")

        # Create a filtered folder to copy the correct images to
        if not os.path.exists(dst_image_folder_path):
            os.makedirs(dst_image_folder_path)

        # Get all files in a list and remove timestamp.txt
        filenames = []
        for (_, _, current_filenames) in os.walk(src_image_folder_path):
            filenames.extend(current_filenames)
            break
        filenames.remove("timestamps.txt")

        # Make sure filenames are sorted in ascending order
        filenames.sort()

        # For every idx copy the corresponding file to the new folder and name it according to the current idx in for loop
        pbar = tqdm(total=len(idx_triples_dict[key]), desc=key)
        for idx, image_idx in enumerate(idx_triples_dict[key]):
            pbar.update(1)
            src_image_filepath = os.path.join(src_image_folder_path,
                                              filenames[image_idx])

            dst_image_filepath = os.path.join(dst_image_folder_path,
                                              str(idx).zfill(8) + ".png")

            shutil.copy(src_image_filepath, dst_image_filepath)

        pbar.close()

        write_reference_timestamps(dst_image_folder_path, reference_timestamps)

        if not keep_orig_image_folders:
            shutil.rmtree(src_image_folder_path)
            os.rename(dst_image_folder_path, src_image_folder_path)


def write_reference_timestamps(dst_image_folder, reference_timestamps):
    print("Update timestamps.txt")
    with open(os.path.join(dst_image_folder, 'timestamps.txt'),
              'w') as filehandle:
        filehandle.writelines("{:.6f}\n".format(timestamp)
                              for timestamp in reference_timestamps)
