import os
from utils.egomotion_compensator import EgomotionCompensator
from utils.utils import *
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.ndimage.filters import uniform_filter1d
from scipy import interpolate
import pymap3d
import open3d as o3d
import itertools


class DataPreprocesser:
    def __init__(self, cfg):
        self.data_folder_path = cfg.data_folder_path

        # "fw_lidar", "mrh_lidar" folders do not have to exist
        self.expected_data_folders = [
            "tf", "gnss", "forward_camera", "right_camera", "left_camera"
        ]
        if not self.check_rosbag_extracted():
            print("Rosbag doesn't seem to have been extracted yet.")
            sys.exit()

        self.keep_orig_data_folders = cfg.keep_orig_data_folders
        self.height, self.width, _ = get_image_size(self.data_folder_path)

        self.raw_gnss_timestamps = None
        self.load_gnss_timestamps()
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
        self.camera_transforms = {"left": np.eye(4), "right": np.eye(4)}
        self.load_camera_transforms()

    def check_rosbag_extracted(self):
        return all(x in os.listdir(self.data_folder_path)
                   for x in self.expected_data_folders)

    def match_data_step_1(self):
        """
        - Matches triples of images with the same time stamp
        - Images with the same timestamp will have the same index
        - A black image is included whenever an image is missing for a specific time stamp
        - Car position and cone positions are also matched to every time stamp
        """

        # Read in the camera timestamp files
        timestamp_arrays_dict = get_camera_timestamps(self.data_folder_path)

        # Initialization
        indices_dict = {
            "forward_camera": [],
            "right_camera": [],
            "left_camera": [],
        }

        # Create large array containing image timestamps, image idx and camera id
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
            # For keeping track to not match images of the same camera
            camera_ids = list(self.camera_id_dict.values())

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

            # Match to GNSS
            gnss_dtimestamps = np.abs(self.raw_gnss_timestamps - ref_timestamp)
            gnss_min_dtimestamp_idx = np.argmin(gnss_dtimestamps)
            gnss_min_dtimestamp = gnss_dtimestamps[gnss_min_dtimestamp_idx]

            if gnss_min_dtimestamp < self.MAX_DTIMESTAMP_GNSS_THRESHOLD:
                self.reference_timestamps.append(ref_timestamp)

            else:
                # If there is no GT position of the car at the timestamp
                idx += 1
                continue

            # Insert first element
            indices_dict[self.id_camera_dict[ref_id]].append(int(ref_idx))
            camera_ids.remove(ref_id)

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

        self.filter_gnss_and_cones()
        self.filter_images(indices_dict)

    def filter_images(self, indices_dict):
        """
        Creates a folder for each camera and makes sure that
        images with the same index correspond to the same timestamp.
        Images are also undistorted here.
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
                if image_idx != -1:
                    src_image_filepath = os.path.join(src_image_folder_path,
                                                      filenames[image_idx])
                    dst_image_filepath = os.path.join(
                        dst_image_folder_path,
                        str(idx).zfill(8) + ".png")

                    shutil.copy(src_image_filepath, dst_image_filepath)

                    undistort_image(self.data_folder_path, dst_image_filepath,
                                    key)
                else:
                    # Create a dummy image
                    dummy_image = np.zeros((self.height, self.width, 3),
                                           np.uint8)
                    dst_image_filepath = os.path.join(
                        dst_image_folder_path,
                        str(idx).zfill(8) + ".png")

                    cv2.imwrite(dst_image_filepath, dummy_image)

                pbar.update(1)

            pbar.close()

            write_reference_timestamps(dst_image_folder_path,
                                       self.reference_timestamps)

            if not self.keep_orig_data_folders:
                shutil.rmtree(src_image_folder_path)

    def match_data_step_2(self, motion_compensation):
        """
        Matches the timewise closest point cloud to each image triple. The result is
        that images and point clouds with the same index in their folders go together.
        """
        # Read in the timestamp files
        timestamp_arrays_dict = get_lidar_timestamps(self.data_folder_path)

        # indices_dict is going to contain one array for each lidar. The array contains a
        # point cloud idx for each reference timestamp
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

        self.filter_point_clouds(indices_dict, motion_compensation)

    def filter_point_clouds(self, indices_dict, motion_compensation):
        """
        Creates a folder for each lidar and makes sure that
        point clouds with the same index correspond to the same timestamp as images
        with that index.
        The images should be matched before to get the reference timestamps.
        Additionally, this function adds empty point clouds when there was
        no point cloud matching a reference time stamp.
        """
        if motion_compensation:
            folder_extension = "_filtered"
        else:
            folder_extension = "_filtered_no_mc"

        for key in indices_dict:
            print("Filtering point clouds in folder {}".format(key))
            src_point_cloud_folder_path = os.path.join(self.data_folder_path,
                                                       key)
            dst_point_cloud_folder_path = os.path.join(self.data_folder_path,
                                                       key + folder_extension)

            # Create a filtered folder to copy the correct point clouds to
            if os.path.exists(dst_point_cloud_folder_path):
                print(
                    "The folder {}{} exist already indicating that the data has already been matched!"
                    .format(key, folder_extension))
                print("{}{} will be removed and the data will be rematched.".
                      format(key, folder_extension))
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
                        self.reference_timestamps[idx], motion_compensation)

                else:
                    # Create an empty point cloud
                    dst_point_cloud_filepath = os.path.join(
                        dst_point_cloud_folder_path,
                        str(idx).zfill(8) + extension)

                    with open(dst_point_cloud_filepath, 'w'):
                        pass

                pbar.update(1)

            pbar.close()

            write_reference_timestamps(dst_point_cloud_folder_path,
                                       self.reference_timestamps)

            if not self.keep_orig_data_folders:
                shutil.rmtree(src_point_cloud_folder_path)

    def load_camera_transforms(self):
        """Open the calibration YAMLs and compute the transform
           from the forward camera to the left and right cameras."""
        calib_left_forward = load_stereo_calib(
            './resources/left.yaml', './resources/left_forward.yaml')
        calib_right_forward = load_stereo_calib(
            './resources/right.yaml', './resources/right_forward.yaml')
        T_left_forward = np.eye(4)
        T_left_forward[:3, :3] = calib_left_forward['rotation_matrix']
        T_left_forward[:3,
                       3] = calib_left_forward['translation_vector'].reshape(
                           (3, ))
        T_right_forward = np.eye(4)
        T_right_forward[:3, :3] = calib_right_forward['rotation_matrix']
        T_right_forward[:3,
                        3] = calib_right_forward['translation_vector'].reshape(
                            (3, ))
        self.camera_transforms["left"] = np.linalg.inv(T_left_forward)
        self.camera_transforms["right"] = np.linalg.inv(T_right_forward)

    def load_gnss_timestamps(self):
        """Fetch GNSS timestamps. Save them as member variables."""
        timestamp_array = []
        with open(os.path.join(self.data_folder_path, "gnss/timestamps.txt"),
                  'r') as timestamps_file:
            for timestamp in timestamps_file:
                timestamp_array.append(timestamp)
            self.raw_gnss_timestamps = np.array(timestamp_array,
                                                dtype=np.float64)

    def interp_gnss(self, ref_timestamp, timestamps, gnss_data):
        """
        Given two timestamps and the gnss data, interpolate
        the gnss data at the ref_timestamp.
        """
        ref_gnss_data = np.zeros(gnss_data[0].shape)
        for data_idx in range(len(ref_gnss_data)):
            f = interpolate.Akima1DInterpolator(timestamps,
                                                gnss_data[:, data_idx])
            ref_gnss_data[data_idx] = f(ref_timestamp)
        return ref_gnss_data

    def parse_gtmd(self, gtmd_path):
        """
        Given path to the GTMD csv file, open the file, parse it
        to produce a numpy array. Each row has the format:
        [Color, Latitude, Longitude, Height, Variance].

        Color column is converted from string to integer: BLUE=0, YELLOW=1.
        """
        gtmd_data = np.genfromtxt(gtmd_path,
                                  delimiter=',',
                                  dtype=None,
                                  encoding=None)
        gtmd_array = np.zeros((len(gtmd_data), 5))
        for idx, gtmd_entry in enumerate(gtmd_data):
            color = gtmd_entry[0]
            if color == 'Blue': gtmd_array[idx, 0] = 0
            elif color == 'Yellow': gtmd_array[idx, 0] = 1
            else: gtmd_array[idx, 0] = 2

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

        GNSS data is formatted as:
        [Latitude, Longitude, Height, INS pitch, INS roll, dual pitch, dual heading]
        """
        ell_wgs84 = pymap3d.Ellipsoid('wgs84')
        long_vehicle, lat_vehicle, h_vehicle = vehicle_gnss[:3]
        INS_pitch, INS_roll, dual_pitch, dual_heading = vehicle_gnss[3:]

        # Only correct for heading
        rotmat = R.from_euler('ZX', [dual_heading + 180, dual_pitch],
                              degrees=True).as_matrix()
        rotmat_enu2fcam = R.from_euler('x', [90], degrees=True).as_matrix()[0]

        # Convert cones to ENU frame of vehicle
        cone_colors, cone_gps = cone_array[:, 0], cone_array[:, 1:4]
        cone_colors = np.expand_dims(cone_colors, axis=1)
        cone_enu, cone_xyz = np.zeros(cone_gps.shape), np.zeros(cone_gps.shape)
        for cone_idx in range(cone_gps.shape[0]):
            cone_enu[cone_idx, :] = pymap3d.geodetic2enu(cone_gps[cone_idx, 0],
                                                         cone_gps[cone_idx, 1],
                                                         cone_gps[cone_idx, 2],
                                                         lat_vehicle,
                                                         long_vehicle,
                                                         h_vehicle,
                                                         ell=ell_wgs84,
                                                         deg=True)

        # Correct cones for rotation, then correct for translation (current vehicle position)
        cone_enu2 = np.transpose(np.matmul(rotmat, np.transpose(cone_enu)))
        cone_enu3 = np.transpose(
            np.matmul(rotmat_enu2fcam, np.transpose(cone_enu2)))
        cone_fcam = cone_enu3

        # Remove cones behind the vehicle, and filter by FOV
        forward_mask = cone_fcam[:, 2] > 0
        cone_colors = cone_colors[forward_mask]
        cone_fcam = cone_fcam[forward_mask, :]
        filtered_cone_array = np.concatenate([cone_colors, cone_fcam], axis=1)

        return filtered_cone_array

    def filter_gnss_and_cones(self):
        """
        After matching image triplets to GNSS timestamps, we must remove the
        non-matched timestamps and corresponding data entries
        """
        print("Filtering GNSS and Cones\n")
        src_gnss_folder_path = os.path.join(self.data_folder_path, "gnss")
        dst_gnss_folder_path = os.path.join(self.data_folder_path,
                                            "car_position_filtered")
        dst_forward_cones_folder_path = os.path.join(self.data_folder_path,
                                                     "forward_cones_filtered")
        dst_left_cones_folder_path = os.path.join(self.data_folder_path,
                                                  "left_cones_filtered")
        dst_right_cones_folder_path = os.path.join(self.data_folder_path,
                                                   "right_cones_filtered")
        os.makedirs(dst_forward_cones_folder_path, exist_ok=True)
        os.makedirs(dst_left_cones_folder_path, exist_ok=True)
        os.makedirs(dst_right_cones_folder_path, exist_ok=True)
        os.makedirs(dst_gnss_folder_path, exist_ok=True)

        # Filter data by interpolation
        init_gnss_data = np.fromfile(
            os.path.join(src_gnss_folder_path, 'init_gnss.bin')).reshape(
                (-1, 7))
        init_gnss_data = init_gnss_data[0]
        gnss_data = np.fromfile(os.path.join(src_gnss_folder_path,
                                             "gnss.bin")).reshape((-1, 7))

        # Preprocess gnss dual heading to avoid wrapping past 0 and 360
        threshold = 300
        for i in range(1, gnss_data.shape[0]):
            diff = gnss_data[i, 6] - gnss_data[i - 1, 6]
            if diff > threshold:
                gnss_data[i, 6] -= 360
            if diff < -threshold:
                gnss_data[i, 6] += 360

        # Smooth out the gnss dual pitch
        gnss_data[:, 5] = uniform_filter1d(gnss_data[:, 5], 20)
        gnss_data[:, 5] -= init_gnss_data[5]

        filtered_gnss_data = []
        for timestamp_idx, ref_timestamp in enumerate(
                self.reference_timestamps):
            gnss_data_ref = self.interp_gnss(ref_timestamp,
                                             self.raw_gnss_timestamps,
                                             gnss_data)
            filtered_gnss_data.append(gnss_data_ref.tolist())

        filtered_gnss_data = np.array(filtered_gnss_data)
        filtered_gnss_data.tofile(
            os.path.join(dst_gnss_folder_path, "gnss.bin"))

        # Fetch cones and process for the forward camera using filtered GNSS data
        gtmd_fn = os.listdir(os.path.join(self.data_folder_path,
                                          '../../gtmd'))[0]
        gtmd_path = os.path.join(self.data_folder_path, '../../gtmd', gtmd_fn)
        cone_array = self.parse_gtmd(gtmd_path)

        for gnss_data_idx in range(filtered_gnss_data.shape[0]):
            forward_cone_array = self.filter_cones(
                cone_array, filtered_gnss_data[gnss_data_idx, :])
            forward_cone_colors, forward_cone_xyz = forward_cone_array[:,
                                                                       0], forward_cone_array[:,
                                                                                              1:]
            forward_cone_colors = forward_cone_colors.reshape((-1, 1))

            # Transform cones to left and right camera
            ones = np.ones((forward_cone_xyz.shape[0], 1))
            homo_cone_array = np.concatenate([forward_cone_xyz, ones], axis=1)
            left_cone_xyz = np.transpose(
                np.matmul(self.camera_transforms["left"],
                          np.transpose(homo_cone_array)))[:, :3]
            left_cone_array = np.concatenate(
                [forward_cone_colors, left_cone_xyz], axis=1)
            right_cone_xyz = np.transpose(
                np.matmul(self.camera_transforms["right"],
                          np.transpose(homo_cone_array)))[:, :3]
            right_cone_array = np.concatenate(
                [forward_cone_colors, right_cone_xyz], axis=1)

            forward_cone_array.tofile(
                os.path.join(dst_forward_cones_folder_path,
                             str(gnss_data_idx).zfill(8) + '.bin'))
            left_cone_array.tofile(
                os.path.join(dst_left_cones_folder_path,
                             str(gnss_data_idx).zfill(8) + '.bin'))
            right_cone_array.tofile(
                os.path.join(dst_right_cones_folder_path,
                             str(gnss_data_idx).zfill(8) + '.bin'))

        if not self.keep_orig_data_folders:
            shutil.rmtree(src_gnss_folder_path)

    def extract_rotations(self):
        # Determine the rotation between the two point clouds using ICP
        # Write to folder called relative rotations and file called relative_rotations.bin

        naming = "relative_rotations"

        print(
            "Calculating relative rotations between consecutive point clouds")

        relative_rotations_folder = os.path.join(self.data_folder_path, naming)

        # Create a folder to store the relative rotations in
        if os.path.exists(relative_rotations_folder):
            print(
                "The folder relative_rotations exist already indicating that they have already been calculated!"
            )
            print(
                "relative_rotations will be removed and the relative rotations will be recalculated."
            )
            shutil.rmtree(relative_rotations_folder)
        os.makedirs(relative_rotations_folder)

        # List containing relative rotations
        relative_rotations = []

        # Choose to use front wing point clouds by default
        # and fall back to mrh point clouds if front wing point clouds
        # are not available
        fw_lidar_folder = os.path.join(self.data_folder_path,
                                       "fw_lidar_filtered")
        mrh_lidar_folder = os.path.join(self.data_folder_path,
                                        "mrh_lidar_filtered")

        if not os.path.exists(fw_lidar_folder) and not os.path.exists(
                mrh_lidar_folder):
            print(
                "Relative rotations could not be extracted. Could not find following paths:\n{}\n{}"
                .format(fw_lidar_folder, mrh_lidar_folder))
            sys.exit()
        elif os.path.exists(fw_lidar_folder):
            which_pc = "fw"
            pc_folder = fw_lidar_folder
        else:
            which_pc = "mrh"
            pc_folder = mrh_lidar_folder

        # Get a list of the point cloud files
        point_cloud_files = os.listdir(pc_folder)
        point_cloud_files.remove('timestamps.txt')
        point_cloud_files = [
            os.path.join(pc_folder, f) for f in point_cloud_files
        ]
        point_cloud_files.sort()

        # Make a list of point cloud file pairs
        point_cloud_file_pairs = np.vstack(
            (np.array(point_cloud_files),
             np.roll(np.array(point_cloud_files), shift=-1)))
        point_cloud_file_pairs = list(
            np.transpose(point_cloud_file_pairs)[:-1])

        pbar = tqdm(total=len(point_cloud_file_pairs), desc=naming)

        # Make a list of initial guesses for ICP
        # Find the reference timestamps that correspond to the point clouds
        if not self.reference_timestamps:
            lidar_filepath_dict = {
                which_pc: os.path.join(pc_folder, "timestamps.txt")
            }
            lidar_timestamps_array_dict = read_timestamps(lidar_filepath_dict)
            self.reference_timestamps = lidar_timestamps_array_dict[which_pc]

        # Calculate initial guesses for the transformations between consecutive
        # point clouds
        timestamps_start = self.reference_timestamps
        timestamps_end = np.roll(self.reference_timestamps, shift=-1)
        timestamp_pairs = np.transpose(
            np.vstack((timestamps_start, timestamps_end)))[:-1]
        initial_guesses = self.egomotion_compensator.get_transformations(
            timestamp_pairs)

        MAX_CHANNEL = 20
        MIN_DIST = 2

        for point_cloud_file_pair, initial_guess in zip(
                point_cloud_file_pairs, initial_guesses):
            fw_pc_1 = read_point_cloud(point_cloud_file_pair[0])

            channels_fw_pc_1 = fw_pc_1[:, 5]
            channels_mask_fw_pc_1 = channels_fw_pc_1 < MAX_CHANNEL

            dist_fw_pc_1 = np.linalg.norm(fw_pc_1[:, :3], axis=1)
            dist_mask_fw_pc_1 = dist_fw_pc_1 > MIN_DIST

            fw_pc_1 = fw_pc_1[np.logical_and(channels_mask_fw_pc_1,
                                             dist_mask_fw_pc_1)]

            fw_pcd_1 = o3d.geometry.PointCloud()
            fw_pcd_1.points = o3d.utility.Vector3dVector(fw_pc_1[:, :3])

            fw_pc_2 = read_point_cloud(point_cloud_file_pair[1])

            channels_fw_pc_2 = fw_pc_2[:, 5]
            channels_mask_fw_pc_2 = channels_fw_pc_2 < MAX_CHANNEL

            dist_fw_pc_2 = np.linalg.norm(fw_pc_2[:, :3], axis=1)
            dist_mask_fw_pc_2 = dist_fw_pc_2 > MIN_DIST

            fw_pc_2 = fw_pc_2[np.logical_and(channels_mask_fw_pc_2,
                                             dist_mask_fw_pc_2)]

            fw_pcd_2 = o3d.geometry.PointCloud()
            fw_pcd_2.points = o3d.utility.Vector3dVector(fw_pc_2[:, :3])

            threshold = 1

            reg_p2p = o3d.registration.registration_icp(
                fw_pcd_1, fw_pcd_2, threshold, initial_guess,
                o3d.registration.TransformationEstimationPointToPoint(),
                o3d.registration.ICPConvergenceCriteria(max_iteration=2000))

            r = R.from_matrix(reg_p2p.transformation[:3, :3])

            evaluation = o3d.registration.evaluate_registration(
                fw_pcd_1, fw_pcd_2, threshold, reg_p2p.transformation)

            eval_metrics = [
                evaluation.fitness, evaluation.inlier_rmse,
                np.asarray(evaluation.correspondence_set).shape[0]
            ]
            rotation = list(r.as_euler('zyx', degrees=True))
            rotation.extend(eval_metrics)

            relative_rotations.append(rotation)

            pbar.update(1)

        pbar.close()

        # Write to file
        with open(os.path.join(relative_rotations_folder, naming + '.txt'),
                  'w') as filehandle:
            # Write a header explaining the data
            filehandle.writelines(
                "yaw, pitch, roll, icp_fitness, inlier_rmse, correspondence_set\n"
            )

            filehandle.writelines(
                "{:.6f}, {:.6f}, {:.6f}, {}, {}, {}\n".format(
                    rel_rot[0],
                    rel_rot[1],
                    rel_rot[2],
                    rel_rot[3],
                    rel_rot[4],
                    rel_rot[5],
                ) for rel_rot in relative_rotations)

    def generate_dim(self):
        """
        Generates the DIM layers given the current data folder
        """
        def concatenate_pcs(pc_files):
            """
            Reads the point clouds provided by a numpy array of point
            cloud file paths
            """
            pc = np.zeros((0, 6))

            for pc_file in pc_files.tolist():
                curr_pc = np.fromfile(pc_file, dtype=np.float64).reshape(-1, 6)
                pc = np.vstack((pc, curr_pc))

            return pc

        def depth_color(points, min_d=0, max_d=30):
            """
            Print Color(HSV's H value) corresponding to distance(m)
            close distance = red , far distance = blue
            """
            dist = np.sqrt(
                np.add(np.power(points[:, 0], 2), np.power(points[:, 1], 2),
                       np.power(points[:, 2], 2)))
            np.clip(dist, 0, max_d, out=dist)

            return (((dist - min_d) / (max_d - min_d)) * 128).astype(np.uint8)

        def project(points, image, R, t, K, debug=False):
            '''
            Project points onto the image and generate the DI and M layers
            '''

            intensities = points[:, 3]
            points = points[:, :3]

            pixels, pixel_filter = project_points_to_pixels(
                points, R, t, K, image.shape)

            # Filter points, pixels and intensities
            points = points[pixel_filter]
            pixels = np.int32(pixels[pixel_filter])
            intensities = intensities[pixel_filter]
            depth = np.sqrt(np.sum(np.power(points, 2), axis=1))

            depth_layer = np.zeros(image.shape[:-1], dtype=np.float16)
            intensity_layer = np.zeros(image.shape[:-1], dtype=np.float16)

            if debug:
                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                color = depth_color(points)

            for i in range(pixels.shape[0]):
                if debug:
                    cv2.circle(
                        hsv_image,
                        (np.int32(pixels[i, 0]), np.int32(pixels[i, 1])), 2,
                        (int(color[i]), 255, 255), -1)

                if depth[i] > depth_layer[pixels[i, 1], pixels[i, 0]]:
                    depth_layer[pixels[i, 1], pixels[i, 0]] = depth[i]
                    intensity_layer[pixels[i, 1], pixels[i,
                                                         0]] = intensities[i]

            mask_layer = (depth_layer > 0).astype(np.bool)

            if debug:
                projection = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
                cv2.imshow('depth', projection)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return depth_layer, mask_layer, intensity_layer

        cameras = ["forward", "left", "right"]
        lidars = ["fw", "mrh"]

        static_transformations_folder = os.path.join(self.data_folder_path,
                                                     "..", "..",
                                                     "static_transformations")

        T_mrh_cam_dict = get_mrh_cam_transformations(
            static_transformations_folder, cameras)
        K_dict = get_intrinsics(static_transformations_folder, cameras)
        pc_files_dict = get_pc_files(self.data_folder_path, lidars)
        img_files_dict = get_img_files(self.data_folder_path, cameras)

        for camera in cameras:

            dst_di_folder_path = os.path.join(self.data_folder_path,
                                              "{}_di".format(camera))
            dst_m_folder_path = os.path.join(self.data_folder_path,
                                             "{}_m".format(camera))
            if os.path.exists(dst_di_folder_path):
                print(
                    "The folder {}_di exists already indicating that the data has already been extracted!"
                    .format(camera))
                print(
                    "{}_di will be removed and the data will be reextracted.".
                    format(camera))
                shutil.rmtree(dst_di_folder_path)
            os.makedirs(dst_di_folder_path)

            if os.path.exists(dst_m_folder_path):
                print(
                    "The folder {}_m exist already indicating that the data has already been extracted!"
                    .format(camera))
                print("{}_m will be removed and the data will be reextracted.".
                      format(camera))
                shutil.rmtree(dst_m_folder_path)
            os.makedirs(dst_m_folder_path)

            K = K_dict[camera]
            T_mrh_cam = T_mrh_cam_dict[camera]
            img_files = img_files_dict[camera]
            pc_files = np.array([
                list(pc_files_dict.items())[x][1]
                for x in range(len(pc_files_dict.keys()))
            ])

            pbar = tqdm(total=len(img_files),
                        desc="DMI of {}_camera".format(camera))

            for idx, img_file in enumerate(img_files):

                pc = concatenate_pcs(pc_files[:, idx])[:, :4]
                img = cv2.imread(img_file)

                d, m, i = project(pc, img, T_mrh_cam[:3, :3], T_mrh_cam[:3, 3],
                                  K)

                di = np.stack((d, i), axis=2)

                di.tofile(
                    os.path.join(dst_di_folder_path,
                                 str(idx).zfill(8) + ".bin"))
                m.tofile(
                    os.path.join(dst_m_folder_path,
                                 str(idx).zfill(8) + ".bin"))

                pbar.update(1)

            pbar.close()
