import os
from utils.egomotion_compensator import *
from utils.utils import *
import numpy as np
from utils.static_transforms import *
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy import interpolate


class DataPreprocesser:
    def __init__(self, cfg):
        self.data_folder_path = cfg.data_folder_path
        self.keep_orig_data_folders = cfg.keep_orig_data_folders
        self.height, self.width, _ = get_image_size(self.data_folder_path)

        self.reference_timestamps = []

        # Point cloud motion compensation
        self.egomotion_compensator = EgomotionCompensator(
            self.data_folder_path)

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
        """
        Matches triples of images with the same time stamp. This function 
        makes sure that images with the same time stamp also have the 
        same index. Unlike match_images_2, this function discards a time
        stamp when not all images are available.
        """
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
        """
        Matches triples of images with the same time stamp. This function 
        makes sure that images with the same time stamp also have the 
        same index. This function includes a black image whenever an image
        is missing for a specific time stamp.
        """

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
        """
        Creates a folder for each camera and makes sure that
        images with the same index correspond to the same timestamp.
        """
        for key in indices_dict:
            print("Filtering images in folder {}".format(key))
            src_image_folder_path = os.path.join(self.data_folder_path, key)
            dst_image_folder_path = os.path.join(self.data_folder_path,
                                                 key + "_filtered")

            # Create a filtered folder to copy the correct images to
            if os.path.exists(dst_image_folder_path):
                print(
                    "The folder {}_filtered exist already indicating that the data has already been matched!"
                    .format(key))
                print(
                    "{}_filtered will be removed and the data will be rematched."
                    .format(key))
                shutil.rmtree(dst_image_folder_path)
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

            if not self.keep_orig_data_folders:
                shutil.rmtree(src_image_folder_path)
                os.rename(dst_image_folder_path, src_image_folder_path)

    def filter_images_2(self, indices_dict):
        """
        Creates a folder for each camera and makes sure that
        images with the same index correspond to the same timestamp.
        Additionally, this function adds dummy images.
        """
        for key in indices_dict:
            print("Filtering images in folder {}".format(key))
            src_image_folder_path = os.path.join(self.data_folder_path, key)
            dst_image_folder_path = os.path.join(self.data_folder_path,
                                                 key + "_filtered")

            # Create a filtered folder to copy the correct images to
            if os.path.exists(dst_image_folder_path):
                print(
                    "The folder {}_filtered exist already indicating that the data has already been matched!"
                    .format(key))
                print(
                    "{}_filtered will be removed and the data will be rematched."
                    .format(key))
                shutil.rmtree(dst_image_folder_path)
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

            if not self.keep_orig_data_folders:
                shutil.rmtree(src_image_folder_path)
                os.rename(dst_image_folder_path, src_image_folder_path)

    def match_point_clouds(self):
        """
        Matches the timewise closest point cloud to each image triple. The result is
        that images and point clouds with the same index in their folders go together.
        """
        # Read in the timestamp files
        timestamp_arrays_dict = get_lidar_timestamps(self.data_folder_path)

        indices_dict = {}
        # Initialization
        for key in timestamp_arrays_dict.keys():
            indices_dict[key] = []

        for key, pc_timestamps in timestamp_arrays_dict.items():

            for ref_timestamp in self.reference_timestamps:
                # The pc_timestamps are the timestamps of the start of the
                # recording of a point cloud. Instead of taking this time
                # stamp, we add 0.05s to approximate the mean time stamp
                # of all the points in the point cloud
                timediff = (pc_timestamps + 0.05) - ref_timestamp
                idx = np.abs(timediff).argmin()

                if np.abs(timediff[idx]) > 0.1:
                    indices_dict[key].append(-1)
                else:
                    indices_dict[key].append(idx)

        self.filter_point_clouds(indices_dict)

    def filter_point_clouds(self, indices_dict):
        """
        Creates a folder for each lidar and makes sure that
        point clouds with the same index correspond to the same timestamp.
        The images should be matched before to get the reference timestamps.
        Additionally, this function adds empty point clouds when there was
        not point cloud matching a reference time stamp.
        """
        for key in indices_dict:
            print("Filtering point clouds in folder {}".format(key))
            src_point_cloud_folder_path = os.path.join(self.data_folder_path,
                                                       key)
            dst_point_cloud_folder_path = os.path.join(self.data_folder_path,
                                                       key + "_filtered")

            # Create a filtered folder to copy the correct point clouds to
            if os.path.exists(dst_point_cloud_folder_path):
                print(
                    "The folder {}_filtered exist already indicating that the data has already been matched!"
                    .format(key))
                print(
                    "{}_filtered will be removed and the data will be rematched."
                    .format(key))
                shutil.rmtree(dst_point_cloud_folder_path)
            os.makedirs(dst_point_cloud_folder_path)

            # Get all files in a list and remove timestamp.txt
            filenames = []
            for (_, _,
                 current_filenames) in os.walk(src_point_cloud_folder_path):
                filenames.extend(current_filenames)
                break
            filenames.remove("timestamps.txt")

            # Find the format that the point clouds are stored in [".npy", ".bin"]
            _, extension = os.path.splitext(filenames[0])

            # Make sure filenames are sorted in ascending order
            filenames.sort()

            # For every idx copy the corresponding file to the new folder and name it according to the current idx in for loop
            pbar = tqdm(total=len(indices_dict[key]), desc=key)
            for idx, point_cloud_idx in enumerate(indices_dict[key]):
                pbar.update(1)
                if point_cloud_idx != -1:
                    src_point_cloud_filepath = os.path.join(
                        src_point_cloud_folder_path,
                        filenames[point_cloud_idx])
                    dst_point_cloud_filepath = os.path.join(
                        dst_point_cloud_folder_path,
                        str(idx).zfill(8) + extension)

                    shutil.copy(src_point_cloud_filepath,
                                dst_point_cloud_filepath)

                    self.egomotion_compensator.egomotion_compensation(
                        dst_point_cloud_filepath, key,
                        self.reference_timestamps[idx])
                else:
                    # Create an empty point cloud
                    dst_point_cloud_filepath = os.path.join(
                        dst_point_cloud_folder_path,
                        str(idx).zfill(8) + extension)

                    with open(dst_point_cloud_filepath, 'w'):
                        pass

            pbar.close()

            write_reference_timestamps(dst_point_cloud_folder_path,
                                       self.reference_timestamps)

            if not self.keep_orig_data_folders:
                shutil.rmtree(src_point_cloud_folder_path)
                os.rename(dst_point_cloud_folder_path,
                          src_point_cloud_folder_path)
