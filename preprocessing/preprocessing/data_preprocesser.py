import os
from utils.utils import *
import numpy as np


class DataPreprocesser:
    def __init__(self, cfg):
        self.data_folder_path = cfg.data_folder_path
        self.keep_orig_image_folders = cfg.keep_orig_image_folders
        self.height, self.width, _ = get_image_size(self.data_folder_path)

        self.reference_timestamps = []

        # Constants
        self.MAX_DTIMESTAMP_THRESHOLD = 0.001

        self.camera_id_dict = {
            'forward_camera': 0,
            'right_camera': 1,
            'left_camera': 2
        }
        self.id_camera_dict = {
            0: 'forward_camera',
            1: 'right_camera',
            2: 'left_camera'
        }

    def match_images_1(self):
        # Read in the timestamp files
        timestamp_arrays_dict = get_camera_timestamps(self.data_folder_path)

        # For every forward camera image timestamp, find the closest
        # timestamps from the left and right camera images
        indices_dict = {
            "forward_camera": [],
            "right_camera": [],
            "left_camera": [],
        }

        for forward_camera_timestamp_idx, forward_camera_timestamp in enumerate(
                timestamp_arrays_dict["forward_camera"]):

            left_camera_dtimestamps = np.abs(
                timestamp_arrays_dict["left_camera"] -
                forward_camera_timestamp)
            left_camera_min_dtimestamp_idx = np.argmin(left_camera_dtimestamps)
            left_camera_min_dtimestamp = left_camera_dtimestamps[
                left_camera_min_dtimestamp_idx]

            right_camera_dtimestamps = np.abs(
                timestamp_arrays_dict["right_camera"] -
                forward_camera_timestamp)
            right_camera_min_dtimestamp_idx = np.argmin(
                right_camera_dtimestamps)
            right_camera_min_dtimestamp = right_camera_dtimestamps[
                right_camera_min_dtimestamp_idx]

            if left_camera_min_dtimestamp < self.MAX_DTIMESTAMP_THRESHOLD and right_camera_min_dtimestamp < self.MAX_DTIMESTAMP_THRESHOLD:

                left_camera_timestamp = timestamp_arrays_dict["left_camera"][
                    left_camera_min_dtimestamp_idx]
                right_camera_timestamp = timestamp_arrays_dict["right_camera"][
                    right_camera_min_dtimestamp_idx]

                if timestamps_within_interval(
                        self.MAX_DTIMESTAMP_THRESHOLD,
                    (forward_camera_timestamp, left_camera_timestamp,
                     right_camera_timestamp)):

                    # Add the indices of the valid triple
                    indices_dict["forward_camera"].append(
                        forward_camera_timestamp_idx)
                    indices_dict["right_camera"].append(
                        right_camera_min_dtimestamp_idx)
                    indices_dict["left_camera"].append(
                        left_camera_min_dtimestamp_idx)

                    self.reference_timestamps.append(
                        np.mean([
                            forward_camera_timestamp, left_camera_timestamp,
                            right_camera_timestamp
                        ]))

        self.filter_images_1(indices_dict)

    def match_images_2(self):

        # Read in the timestamp files
        timestamp_arrays_dict = get_camera_timestamps(self.data_folder_path)

        indices_dict = {}
        # Initialization
        for key in timestamp_arrays_dict.keys():
            indices_dict[key] = []

        timestamp_idx_id_array = np.empty((0, 3), dtype=np.float64)

        for key, timestamps in timestamp_arrays_dict.items():
            indices = np.arange(timestamps.shape[0], dtype=np.float64)
            camera_ids = np.ones(timestamps.shape[0],
                                 dtype=np.float64) * self.camera_id_dict[key]
            temp_array = np.vstack((timestamps, indices, camera_ids))
            timestamp_idx_id_array = np.vstack(
                (timestamp_idx_id_array, np.transpose(temp_array)))

        # Sort timestamp_idx_id_array according to the timestamps
        sorting_indices = timestamp_idx_id_array[:, 0].argsort()
        timestamp_idx_id_array = timestamp_idx_id_array[sorting_indices]

        idx = 0
        incomplete_data_counter = 0
        while idx < timestamp_idx_id_array.shape[0]:
            camera_ids = [0, 1, 2]

            # First element
            ref_timestamp = timestamp_idx_id_array[idx, 0]
            ref_idx = timestamp_idx_id_array[idx, 1]
            ref_id = timestamp_idx_id_array[idx, 2]
            # Second element
            sec_timestamp = timestamp_idx_id_array[idx + 1, 0]
            sec_idx = timestamp_idx_id_array[idx + 1, 1]
            sec_id = timestamp_idx_id_array[idx + 1, 2]
            # Third element
            thi_timestamp = timestamp_idx_id_array[idx + 2, 0]
            thi_idx = timestamp_idx_id_array[idx + 2, 1]
            thi_id = timestamp_idx_id_array[idx + 2, 2]

            # Insert first element
            indices_dict[self.id_camera_dict[ref_id]].append(int(ref_idx))
            camera_ids.remove(ref_id)

            # Append timestamp
            self.reference_timestamps.append(ref_timestamp)

            # Insert second element
            if sec_timestamp - ref_timestamp < self.MAX_DTIMESTAMP_THRESHOLD:
                if ref_id != sec_id:
                    # Insert second element
                    indices_dict[self.id_camera_dict[sec_id]].append(
                        int(sec_idx))
                    camera_ids.remove(sec_id)
                else:
                    # Two consecutive timestamps of the same camera withing MAX_DTIMESTAMP_THRESHOLD
                    # We should never reach here..
                    idx += 2
                    incomplete_data_counter += 1
                    continue
            else:
                # In this case, we only insert the first element's idx and two fake indices
                for camera_id in camera_ids:
                    indices_dict[self.id_camera_dict[camera_id]].append(
                        int(-1))
                idx += 1
                incomplete_data_counter += 1
                continue

            # Insert third element
            if thi_timestamp - ref_timestamp < self.MAX_DTIMESTAMP_THRESHOLD:
                if thi_id == camera_ids[0]:
                    # Insert third element
                    indices_dict[self.id_camera_dict[thi_id]].append(
                        int(thi_idx))
                    camera_ids.remove(thi_id)
                else:
                    # Two consecutive timestamps of the same camera withing MAX_DTIMESTAMP_THRESHOLD
                    # We should never reach here..
                    idx += 3
                    incomplete_data_counter += 1
                    continue
            else:
                # In this case, we only insert the first two elements, the third element's timestamp
                # is too far off
                for camera_id in camera_ids:
                    indices_dict[self.id_camera_dict[camera_id]].append(
                        int(-1))
                idx += 2
                incomplete_data_counter += 1
                continue

            # Update idx to next section of the indices_dict
            idx += 3

        print("Incomplete data pairs {}/{}".format(
            incomplete_data_counter,
            int(timestamp_idx_id_array[-1, 0] - timestamp_idx_id_array[0, 0]) *
            10))

        self.filter_images_2(indices_dict)

    def filter_images_1(self, indices_dict):
        for key in indices_dict:
            print("Filtering images in folder {}".format(key))
            src_image_folder_path = os.path.join(self.data_folder_path, key)
            dst_image_folder_path = os.path.join(self.data_folder_path,
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
            pbar = tqdm(total=len(indices_dict[key]), desc=key)
            for idx, image_idx in enumerate(indices_dict[key]):
                pbar.update(1)
                src_image_filepath = os.path.join(src_image_folder_path,
                                                  filenames[image_idx])

                dst_image_filepath = os.path.join(dst_image_folder_path,
                                                  str(idx).zfill(8) + ".png")

                shutil.copy(src_image_filepath, dst_image_filepath)

            pbar.close()

            write_reference_timestamps(dst_image_folder_path,
                                       self.reference_timestamps)

            if not self.keep_orig_image_folders:
                shutil.rmtree(src_image_folder_path)
                os.rename(dst_image_folder_path, src_image_folder_path)

    def filter_images_2(self, indices_dict):
        for key in indices_dict:
            print("Filtering images in folder {}".format(key))
            src_image_folder_path = os.path.join(self.data_folder_path, key)
            dst_image_folder_path = os.path.join(self.data_folder_path,
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
            pbar = tqdm(total=len(indices_dict[key]), desc=key)
            for idx, image_idx in enumerate(indices_dict[key]):
                pbar.update(1)
                if image_idx != -1:
                    src_image_filepath = os.path.join(src_image_folder_path,
                                                      filenames[image_idx])
                    dst_image_filepath = os.path.join(
                        dst_image_folder_path,
                        str(idx).zfill(8) + ".png")

                    shutil.copy(src_image_filepath, dst_image_filepath)
                else:
                    # Create a dummy image
                    dummy_image = np.zeros((self.height, self.width, 3),
                                           np.uint8)
                    dst_image_filepath = os.path.join(
                        dst_image_folder_path,
                        str(idx).zfill(8) + ".png")

                    cv2.imwrite(dst_image_filepath, dummy_image)

            pbar.close()

            write_reference_timestamps(dst_image_folder_path,
                                       self.reference_timestamps)

            if not self.keep_orig_image_folders:
                shutil.rmtree(src_image_folder_path)
                os.rename(dst_image_folder_path, src_image_folder_path)