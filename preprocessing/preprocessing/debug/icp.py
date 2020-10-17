import os
import sys
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import copy
import itertools
from utils.utils import *
from scipy.spatial.transform import Slerp
from scipy import interpolate


def read_point_cloud(point_cloud_file):
    _, ext = os.path.splitext(point_cloud_file)

    if ext == '.bin':
        point_cloud = (np.fromfile(point_cloud_file,
                                   dtype=np.float64).reshape(-1, 6))
    elif ext == '.npy':
        point_cloud = (np.load(point_cloud_file,
                               dtype=np.float64).reshape(-1, 6))
    else:
        print("Invalid point cloud format encountered.")
        sys.exit()

    return point_cloud


def draw_registration_result(source, target, transformation, window_name):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      window_name=window_name)


def get_initial_guesses():

    data_folder_path = "/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-07-12_tuggen/data/autocross_2020-07-12-09-54-31"

    T = read_static_transformations(data_folder_path)

    ego_wor_transformations = read_dynamic_transformation(
        "egomotion_to_world", data_folder_path)

    # Create objects that can be queried for transformations at specific timestamps
    key_rots = R.from_quat(ego_wor_transformations[:, 4:8])
    key_times = ego_wor_transformations[:, 0]
    R_slerp_ego_wor = Slerp(key_times, key_rots)
    t_interpolator_ego_wor = interpolate.interp1d(key_times,
                                                  ego_wor_transformations[:,
                                                                          1:4],
                                                  axis=0,
                                                  assume_sorted=True)

    # Get the timestamps that correspond to the point clouds
    which_pc = "fw"
    pc_folder = "/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-07-12_tuggen/data/autocross_2020-07-12-09-54-31/fw_lidar_filtered"
    lidar_filepath_dict = {which_pc: os.path.join(pc_folder, "timestamps.txt")}
    lidar_timestamps_array_dict = read_timestamps(lidar_filepath_dict)
    reference_timestamps = lidar_timestamps_array_dict[which_pc]

    # Create timestamp tuples
    timestamps_start = reference_timestamps
    timestamps_end = np.roll(reference_timestamps, shift=-1)
    timestamp_tuples = np.transpose(
        np.vstack((timestamps_start, timestamps_end)))[:-1]

    # Calculate the transformations T that correspond to the timestamp tuples
    T_mrh_ego = T["mrh_lidar_to_egomotion"]

    T_ego_wor = np.zeros((timestamp_tuples.shape[0], 4, 4))
    start_rots = R_slerp_ego_wor(timestamp_tuples[:, 0]).as_matrix()
    start_trans = t_interpolator_ego_wor(timestamp_tuples[:, 0])
    T_ego_wor[:, :3, :3] = start_rots
    T_ego_wor[:, :3, 3] = start_trans
    T_ego_wor[:, 3, 3] = 1

    T_wor_ego = np.zeros((timestamp_tuples.shape[0], 4, 4))
    end_rots = R_slerp_ego_wor(timestamp_tuples[:, 1]).as_matrix()
    end_trans = t_interpolator_ego_wor(timestamp_tuples[:, 1])
    T_wor_ego[:, :3, :3] = end_rots
    T_wor_ego[:, :3, 3] = end_trans
    T_wor_ego[:, 3, 3] = 1
    T_wor_ego = np.linalg.inv(T_wor_ego)

    T_ego_mrh = np.linalg.inv(T_mrh_ego)

    Ts = T_ego_mrh @ T_wor_ego @ T_ego_wor @ T_mrh_ego

    return Ts


# Read in point cloud files and put them into a list of tuples
base_folder = "/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-07-12_tuggen/data/autocross_2020-07-12-09-54-31/fw_lidar_filtered"

files = os.listdir(base_folder)
files.remove('timestamps.txt')
files = [os.path.join(base_folder, f) for f in files]

files.sort()

point_cloud_pairs = list(
    np.transpose(
        np.vstack((np.array(files), np.roll(np.array(files), shift=-1))))[:-1])

# Get initial guesses also in form of of a list
initial_guesses = get_initial_guesses()

# OUTPUT: list of transformations found using ICP
relative_trafo = []

METHOD = 0
USE_CHANNELS = True
MAX_COUNT = 300
MIN_COUNT = 0
MAX_CHANNEL = 20
MIN_DIST = 2
counter = 0
for pair, initial_guess in zip(point_cloud_pairs, initial_guesses):

    if counter > MAX_COUNT:
        break

    if counter < MIN_COUNT:
        counter += 1
        continue

    fw_pc_1 = read_point_cloud(pair[0])

    channels_fw_pc_1 = fw_pc_1[:, 5]
    channels_mask_fw_pc_1 = channels_fw_pc_1 < MAX_CHANNEL

    dist_fw_pc_1 = np.linalg.norm(fw_pc_1[:, :3], axis=1)
    dist_mask_fw_pc_1 = dist_fw_pc_1 > MIN_DIST

    fw_pc_1 = fw_pc_1[np.logical_and(channels_mask_fw_pc_1, dist_mask_fw_pc_1)]

    fw_pc_2 = read_point_cloud(pair[1])
    channels_fw_pc_2 = fw_pc_2[:, 5]
    channels_mask_fw_pc_2 = channels_fw_pc_2 < MAX_CHANNEL

    dist_fw_pc_2 = np.linalg.norm(fw_pc_2[:, :3], axis=1)
    dist_mask_fw_pc_2 = dist_fw_pc_2 > MIN_DIST

    fw_pc_2 = fw_pc_2[np.logical_and(channels_mask_fw_pc_2, dist_mask_fw_pc_2)]

    #####
    fw_pcd_1 = o3d.geometry.PointCloud()
    fw_pcd_1.points = o3d.utility.Vector3dVector(fw_pc_1[:, :3])

    # plane_model, inliers_1 = fw_pcd_1.segment_plane(distance_threshold=0.2,
    #                                         ransac_n=3,
    #                                         num_iterations=2000)
    # fw_pcd_1 = fw_pcd_1.select_by_index(inliers_1, invert=True)

    fw_pcd_2 = o3d.geometry.PointCloud()
    fw_pcd_2.points = o3d.utility.Vector3dVector(fw_pc_2[:, :3])

    # print("Arrays are equal: {}".format(np.array_equal(np.array(fw_pc_1[:, :3]), np.array(fw_pc_2[:, :3]))))
    # print(np.array(fw_pc_1[3, :3]))
    # print(np.array(fw_pc_2[3, :3]))
    # print("Pair: {}".format(pair))

    # plane_model, inliers_2 = fw_pcd_2.segment_plane(distance_threshold=0.2,
    #                                         ransac_n=3,
    #                                         num_iterations=2000)
    # fw_pcd_2 = fw_pcd_2.select_by_index(inliers_2, invert=True)

    # draw_registration_result(fw_pcd_1, fw_pcd_2, np.eye(4), "np.eye(4)")
    # draw_registration_result(fw_pcd_1, fw_pcd_2, initial_guess, "TF")

    threshold = 1

    # reg_p2p = o3d.registration.registration_icp(
    #     fw_pcd_1, fw_pcd_2, threshold, np.eye(4),
    #     o3d.registration.TransformationEstimationPointToPoint(),
    #     o3d.registration.ICPConvergenceCriteria(max_iteration=5000))
    if METHOD == 0:
        reg_p2p_init = o3d.registration.registration_icp(
            fw_pcd_1, fw_pcd_2, threshold, initial_guess,
            o3d.registration.TransformationEstimationPointToPoint(),
            o3d.registration.ICPConvergenceCriteria(
                relative_fitness=1.000000e-08,
                relative_rmse=1.000000e-08,
                max_iteration=2000))
    elif METHOD == 1:
        reg_p2p_init = o3d.registration.registration_ransac_based_on_correspondence(
            fw_pcd_1,
            fw_pcd_2,
            o3d.registration.CorrespondenceCheckerBasedOnDistance(2),
            max_correspondence_distance=threshold)

    # draw_registration_result(fw_pcd_1, fw_pcd_2, reg_p2p.transformation,
    #                          "np.eye(4)")
    # draw_registration_result(fw_pcd_1, fw_pcd_2, reg_p2p_init.transformation,
    #                          "TF")

    r = R.from_matrix(reg_p2p_init.transformation[:3, :3])
    evaluation = o3d.registration.evaluate_registration(
        fw_pcd_1, fw_pcd_2, threshold, reg_p2p_init.transformation)
    eval_metrics = [
        evaluation.fitness, evaluation.inlier_rmse,
        np.asarray(evaluation.correspondence_set).shape[0]
    ]
    rot = list(r.as_euler('zyx', degrees=True))
    rot.extend(eval_metrics)

    relative_trafo.append(rot)
    print(rot)

    counter += 1