import os
from utils.egomotion_compensator import *
from utils.utils import *
import numpy as np
import sys
from utils.static_transforms import *
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy import interpolate
import pymap3d


class DataPreprocesser:
    def __init__(self, cfg):
        self.data_folder_path = cfg.data_folder_path

        self.expected_data_folders = [
            "tf", "gnss", "fw_lidar", "mrh_lidar", "forward_camera",
            "right_camera", "left_camera"
        ]
        if not self.check_rosbag_extracted():
            print("Rosbag doesn't seem to have been extracted yet.")
            sys.exit()

        self.keep_orig_data_folders = cfg.keep_orig_data_folders
        self.height, self.width, _ = get_image_size(self.data_folder_path)

        self.raw_gnss_timestamps = None
        self.load_gnss()
        self.reference_timestamps = []

        # Point cloud motion compensation
        self.egomotion_compensator = EgomotionCompensator(
            self.data_folder_path)

        # Constants
        self.MAX_DTIMESTAMP_THRESHOLD = 0.001
        self.MAX_DTIMESTAMP_GNSS_THRESHOLD = 0.05

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

    def check_rosbag_extracted(self):
        return all(x in os.listdir(self.data_folder_path)
                   for x in self.expected_data_folders)

    def load_gnss(self):
        """
        Matches each gnss message (20 Hz) to a group of images and pointclouds
        (with corresponding timestamps).

        Determine 3D position of visible cones at each matched timestamp.
        Write parsed cone positions and timestamps to a new data folder.
        """

        # Fetch GNSS timestamps
        timestamp_array = []
        with open(os.path.join(self.data_folder_path, "gnss/timestamps.txt"), 'r') as timestamps_file:
            for timestamp in timestamps_file:
                timestamp_array.append(timestamp)
            self.raw_gnss_timestamps = np.array(timestamp_array, dtype=np.float)

    def interp_gnss(self, ref_timestamp, timestamps, gnss_data):
        """
        Given two timestamps and two gnss data points, interpolate
        the gnss data at the ref_timestamp.
        """
        ref_gnss_data = np.zeros(gnss_data[0].shape)
        for data_idx in range(len(ref_gnss_data)):
            f = interpolate.interp1d(timestamps,
                                     [gnss_data[0][data_idx], gnss_data[1][data_idx]],
                                     fill_value='extrapolate')
            ref_gnss_data[data_idx] = f(ref_timestamp)
        return ref_gnss_data

    def parse_gtmd(self, gtmd_path):
        """
        Given path to the GTMD csv file, open the file, parse it
        to produce a numpy array. Each row has the format:
        [Color, Latitude, Longitude, Height, Variance].

        Color column is converted from string to integer: BLUE=0, YELLOW=1.
        """
        gtmd_data = np.genfromtxt(gtmd_path, delimiter=',', dtype=None)
        gtmd_array = np.zeros((len(gtmd_data), 5))
        for idx, gtmd_entry in enumerate(gtmd_data):
            # Parse Color
            color = gtmd_entry[0].decode("utf-8")
            if color == 'Blue':
                gtmd_array[idx, 0] = 0
            elif color == 'Yellow':
                gtmd_array[idx, 0] = 1
            else:
                gtmd_array[idx, 0] = 2

            # Parse Lat, Long, Height, Variance
            for col_idx in range(1, 5):
                gtmd_array[idx, col_idx] = gtmd_entry[col_idx]
        return gtmd_array

    def filter_cones(self, cone_array, vehicle_gnss):
        """
        Given a cone array (N, 5) where each row is formatted as:
        [Color, Latitude, Longitude, Height, Variance], and the
        gnss data for the vehicle at an instance in time, determine the
        3D position of cones within the vehicle's HFOV at that instance.
        """
        ell_wgs84 = pymap3d.Ellipsoid('wgs84')
        long_vehicle, lat_vehicle, h_vehicle = vehicle_gnss[:3]
        pitch, roll, heading = vehicle_gnss[3:]
        rotmat = R.from_euler('xyz', [pitch, roll, heading], degrees=True).as_matrix()
        rotmat_enu2egomotion = R.from_euler('z', [90], degrees=True).as_matrix()[0]
        transvec_enu2egomotion = np.array([1.166, 0, 0]).reshape((1, 3))
        hfov_half_vehicle = 80

        # Convert cones to ENU
        cone_colors, cone_gps = cone_array[:, 0], cone_array[:, 1:4]
        cone_colors = np.expand_dims(cone_colors, axis=1)
        cone_enu, cone_xyz = np.zeros(cone_gps.shape), np.zeros(cone_gps.shape)
        for cone_idx in range(cone_gps.shape[0]):
            cone_enu[cone_idx, :] = pymap3d.geodetic2enu(lat_vehicle, long_vehicle, h_vehicle,
                                                         cone_gps[cone_idx, 0], cone_gps[cone_idx, 1], cone_gps[cone_idx, 2],
                                                         ell=ell_wgs84, deg=True)
            cone_enu = np.matmul(cone_enu, np.transpose(rotmat))
            cone_xyz = np.matmul(cone_enu, np.transpose(rotmat_enu2egomotion))
            cone_xyz += transvec_enu2egomotion

        # Remove cones behind the vehicle, and filter by FOV
        forward_mask = cone_xyz[:, 0] > 0
        cone_angles = np.rad2deg(np.arctan2(cone_xyz[:, 1], cone_xyz[:, 0]))
        hfov_mask = np.logical_and(cone_angles > -hfov_half_vehicle,
                                   cone_angles < hfov_half_vehicle)
        combined_mask = np.logical_and(forward_mask, hfov_mask)

        # Debug
        # print(f"Initial cones: {cone_array.shape[0]}")
        # print(f"Remaining cones: {np.sum(combined_mask)}")

        cone_colors = cone_colors[combined_mask]
        cone_xyz = cone_xyz[combined_mask, :]
        filtered_cone_array = np.concatenate([cone_colors, cone_xyz], axis=1)

        return filtered_cone_array


    def filter_gnss_and_cones(self, indices):
        """
        After matching image triplets to GNSS timestamps, we must remove the
        non-matched timestamps and corresponding data entries
        """
        print("Filtering gnss")

        # Pathing
        src_gnss_folder_path = os.path.join(self.data_folder_path, "gnss")
        dst_gnss_folder_path = os.path.join(self.data_folder_path, "gnss_filtered")
        dst_cones_folder_path = os.path.join(self.data_folder_path, "cones_filtered")
        os.makedirs(dst_cones_folder_path, exist_ok=True)
        os.makedirs(dst_gnss_folder_path, exist_ok=True)

        # Filter data by interpolation
        gnss_data = np.fromfile(os.path.join(src_gnss_folder_path, "gnss.bin")).reshape((-1, 6))
        filtered_gnss_data = []
        for timestamp_idx, ref_timestamp in enumerate(self.reference_timestamps):
            gnss_data_idx_1 = indices[timestamp_idx]
            gnss_timestamp_1 = self.raw_gnss_timestamps[gnss_data_idx_1]
            gnss_data_1 = gnss_data[gnss_data_idx_1, :]

            # Determine whether to use PREV or NEXT data point for interp
            if (gnss_timestamp_1 > ref_timestamp and gnss_data_idx_1 > 0) or \
            (gnss_data_idx_1 == len(self.raw_gnss_timestamps)-1):
                gnss_data_idx_2 = gnss_data_idx_1-1
                gnss_data_2 = gnss_data[gnss_data_idx_2, :]
                gnss_timestamp_2 = self.raw_gnss_timestamps[gnss_data_idx_2]

            else:
                gnss_data_idx_2 = gnss_data_idx_1+1
                gnss_data_2 = gnss_data[gnss_data_idx_2, :]
                gnss_timestamp_2 = self.raw_gnss_timestamps[gnss_data_idx_2]

            gnss_data_ref = self.interp_gnss(ref_timestamp,
                                             [gnss_timestamp_1, gnss_timestamp_2],
                                             [gnss_data_1, gnss_data_2])
            filtered_gnss_data.append(gnss_data_ref.tolist())

        filtered_gnss_data = np.array(filtered_gnss_data)
        filtered_gnss_data.tofile(os.path.join(dst_gnss_folder_path, "gnss.bin"))

        # Fetch cones and process them using filtered GNSS data
        gtmd_fn = os.listdir(os.path.join(self.data_folder_path, '../../gtmd'))[0]
        gtmd_path = os.path.join(self.data_folder_path, '../../gtmd', gtmd_fn)
        cone_array = self.parse_gtmd(gtmd_path)

        for gnss_data_idx in range(filtered_gnss_data.shape[0]):
            cone_array_local = self.filter_cones(cone_array,
                                                 gnss_data[gnss_data_idx, :])
            cone_array_local.tofile(os.path.join(dst_cones_folder_path, str(gnss_data_idx).zfill(8) + '.bin'))

        if not self.keep_orig_data_folders:
            shutil.rmtree(src_gnss_folder_path)
            os.rename(dst_gnss_folder_path, src_gnss_folder_path)

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
            "gnss": []
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

                    mean_image_timestamp = np.mean([forward_camera_timestamp,
                                                    left_camera_timestamp,
                                                    right_camera_timestamp])

                    gnss_dtimestamps = np.abs(self.raw_gnss_timestamps -
                                              mean_image_timestamp)
                    gnss_min_dtimestamp_idx = np.argmin(gnss_dtimestamps)
                    gnss_min_dtimestamp = gnss_dtimestamps[gnss_min_dtimestamp_idx]

                    if gnss_min_dtimestamp < self.MAX_DTIMESTAMP_GNSS_THRESHOLD:
                        # Add the indices of the valid image triple and gnss
                        indices_dict["forward_camera"].append(
                            forward_camera_timestamp_idx)
                        indices_dict["right_camera"].append(
                            right_camera_min_dtimestamp_idx)
                        indices_dict["left_camera"].append(
                            left_camera_min_dtimestamp_idx)
                        indices_dict["gnss"].append(gnss_min_dtimestamp_idx)

                        self.reference_timestamps.append(mean_image_timestamp)

        self.filter_gnss_and_cones(indices_dict["gnss"])
        del indices_dict["gnss"]
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
        while idx < timestamp_idx_id_array.shape[0] - 2:
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

            # Match to GNSS
            gnss_dtimestamps = np.abs(self.raw_gnss_timestamps -
                                      ref_timestamp)
            gnss_min_dtimestamp_idx = np.argmin(gnss_dtimestamps)
            gnss_min_dtimestamp = gnss_dtimestamps[gnss_min_dtimestamp_idx]

            # TODO: Decide on GNSS or image timestamp as reference?
            if gnss_min_dtimestamp < self.MAX_DTIMESTAMP_GNSS_THRESHOLD:
                indices_dict["gnss"].append(gnss_min_dtimestamp_idx)
                self.reference_timestamps.append(ref_timestamp)

            else:
                continue

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

        self.filter_gnss_and_cones(indices_dict["gnss"])
        del indices_dict["gnss"]
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
                timediff = np.abs((pc_timestamps + 0.05) - ref_timestamp)
                min_idx = timediff.argmin()

                if timediff[min_idx] > 0.1:
                    indices_dict[key].append(-1)
                else:
                    indices_dict[key].append(min_idx)

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
