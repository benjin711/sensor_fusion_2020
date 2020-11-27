import argparse
import os
import numpy as np
import sys
import cv2
import open3d as o3d
import yaml
from matplotlib import pyplot as plt
from glob import glob


def command_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-b',
        '--base_folder',
        default='/media/benjin/Samsung_T5/AMZ/sensor_fusion_data',
        type=str,
        help='Specify local path of the sensor_fusion_data folder')

    parser.add_argument(
        '-l',
        '--label_paths_file',
        default=
        '/home/benjin/Development/git/sensor_fusion_2020/data/full/test.txt',
        type=str,
        help='Specify which data split to calculate metrics on')

    parser.add_argument('-m',
                        '--mode',
                        default='metrics',
                        type=str,
                        choices=['vis, metrics'],
                        help='Specify what the program should do')

    parser.add_argument(
        '-md',
        '--max_distance',
        default=80,
        type=int,
        help=
        'Maximal expected prediction distance. Every prediction above will be discarded.'
    )

    parser.add_argument('-i',
                        '--interval_length',
                        default=2,
                        type=int,
                        help='Interval length for grouping predicted cones.')

    cfg = parser.parse_args()

    return cfg


def check_cfg(cfg):
    if cfg.max_distance % cfg.interval_length != 0:
        print("Command line arguments invalid!")
        sys.exit()


def match_cone_arrays(cfg):
    def duplicate_gt(gt_cone_array_path):
        duplicates = glob(
            gt_cone_array_path.replace('forward',
                                       '*').replace('left',
                                                    '*').replace('right', '*'))
        return len(duplicates) > 0

    def get_lidar_cone_array_path(gt_cone_array_path):
        cameras = ["left", "right", "forward"]
        for camera in cameras:
            gt_cone_array_path = gt_cone_array_path.replace(
                camera + "_cones_corrected", "lidar_cone_arrays_filtered")

        lidar_cone_array_path = gt_cone_array_path

        if os.path.exists(lidar_cone_array_path):
            return lidar_cone_array_path
        else:
            return None

    def get_T_mrh_to_ego(static_transformation_folder):
        T_mrh_ego = np.eye(4)

        T_mrh_ego_file = os.path.join(static_transformations_folder,
                                      "static_transformations.yaml")
        with open(T_mrh_ego_file, "r") as f:
            transformations = yaml.load(f, Loader=yaml.FullLoader)
        for key in transformations.keys():
            rot = transformations[key]['mrh_lidar_to_egomotion']['rotation']
            trans = transformations[key]['mrh_lidar_to_egomotion'][
                'translation']
        rotv = np.array([rot['roll'], rot['pitch'], rot['yaw']])
        R_mrh_ego, _ = cv2.Rodrigues(rotv)
        t_mrh_ego = np.array([trans['x'], trans['y'], trans['z']])
        T_mrh_ego[:3, :3], T_mrh_ego[:3, 3] = R_mrh_ego, t_mrh_ego.reshape(-1)

        return T_mrh_ego

    def get_T_camera_to_mrh(static_transformations_folder, camera):
        T_mrh_camera = np.eye(4)

        T_mrh_camera_file = os.path.join(static_transformations_folder,
                                         "extrinsics_mrh_" + camera + ".yaml")
        transformation_file = cv2.FileStorage(T_mrh_camera_file,
                                              cv2.FILE_STORAGE_READ)
        R_mrh_camera = transformation_file.getNode("R_mtx").mat()
        t_mrh_camera = transformation_file.getNode("t_mtx").mat()
        transformation_file.release()
        T_mrh_camera[:3, :
                     3], T_mrh_camera[:3,
                                      3] = R_mrh_camera, t_mrh_camera.reshape(
                                          -1)
        return np.linalg.inv(T_mrh_camera)

    def extract_test_day_from_path(path, base_folder):
        path_ = path.replace(base_folder, '')
        test_day = os.path.split(path_)[0]
        return test_day

    def extract_camera_from_path(gt_cone_array_path):
        cameras = ["left", "right", "forward"]
        for camera in cameras:
            if gt_cone_array_path.find(camera) != -1:
                return camera

        print("No camera could be extracted from path!")
        sys.exit()

    cone_arrays_dict = {"lidar": [], "gt": []}

    # - [ ]  Read in the label paths from the test.txt
    # - [ ]  Change the paths from label to cones corrected
    with open(cfg.label_paths_file, "r") as f:
        orig_label_paths = f.readlines()

    base_folder_name = os.path.basename(cfg.base_folder)
    index = orig_label_paths[0].find(base_folder_name)
    index += len(base_folder_name) + 1

    gt_cone_array_paths = []
    lidar_cone_array_paths = []
    counter_1, counter_2, counter_3 = 0, 0, 0
    for label_path in orig_label_paths:
        label_path = os.path.join(cfg.base_folder, label_path[index:].rstrip())
        if os.path.exists(label_path):
            gt_cone_array_path = label_path.replace('labels',
                                                    'cones_corrected').replace(
                                                        'txt', 'bin')

            if os.path.exists(gt_cone_array_path
                              ) and not duplicate_gt(gt_cone_array_path):
                # - [ ]  Filter out duplicates, meaning gt cone arrays that belong to the same timestamp
                # - [ ]  For each gt_cone_array, find the corresponding lidar file path, skip if not found
                lidar_cone_array_path = get_lidar_cone_array_path(
                    gt_cone_array_path)
                if lidar_cone_array is not None:
                    gt_cone_array_paths.append(gt_cone_array_path)
                    lidar_cone_array_paths.append(lidar_cone_array_path)
                else:
                    counter_3 += 1
            else:
                counter_2 += 1
        else:
            counter_1 += 1

    if counter_1:
        print(
            "{} specified label files don't exist on the local system. Data bases out of sync?"
            .format(counter_1))
    if counter_2:
        print(
            "{} specified gt cone array files don't exist on the local system. Data bases out of sync?"
            .format(counter_2))

    # All gt cone arrays are in their respective camera frame
    for gt_cone_array_path, lidar_cone_array_path in zip(
            gt_cone_array_paths, lidar_cone_array_paths):
        test_day = extract_test_day_from_path(gt_cone_array_path)

        T_mrh_ego = get_T_mrh_to_ego(
            os.path.join(cfg.base_folder, test_day, "static_transformations"))

        camera = extract_camera_from_path(gt_cone_array_path)

        T_camera_to_mrh = get_T_camera_to_mrh(
            os.path.join(cfg.base_folder, test_day, "static_transformations"),
            camera)

        T_camera_to_ego = T_mrh_ego @ T_camera_to_mrh

        gt_cone_array = np.fromfile(gt_cone_array_file).reshape(-1, 4)
        cone_types = gt_cone_array[:, 0]
        cone_positions = gt_cone_array[:, 1:]

        h_cone_positions = np.hstack(
            (cone_positions, np.ones((cone_positions.shape[0], 1))))
        h_cone_positions = T_camera_ego @ h_cone_positions.T
        gt_cone_array = np.hstack(
            (cone_types.reshape(-1, 1), h_cone_positions.T[:, :3]))
        cone_arrays_dict["gt"].append(gt_cone_array)

        lidar_cone_array = np.fromfile(lidar_cone_array_file).reshape(-1, 3)
        cone_arrays_dict["lidar"].append(lidar_cone_array)

    # Get ground truth cone arrays in ego motion frame
    # Use the GT cone arrays from the left and right camera frames
    # gt_cone_arrays_dict = {"left": {}, "right": {}, "forward": {}}
    # indices = []

    # # Pathing
    # label_paths_file = cfg.label_paths_file[0]
    # static_transformations_folder = os.path.join(label_paths_file, "..", "..",
    #                                              "static_transformations")

    # # Get static transformation: mrh to ego

    # for camera in gt_cone_arrays_dict.keys():
    #     # Get static transformation: camera to mrh
    #     T_mrh_camera = np.eye(4)

    #     T_mrh_camera_file = os.path.join(static_transformations_folder,
    #                                      "extrinsics_mrh_" + camera + ".yaml")
    #     transformation_file = cv2.FileStorage(T_mrh_camera_file,
    #                                           cv2.FILE_STORAGE_READ)
    #     R_mrh_camera = transformation_file.getNode("R_mtx").mat()
    #     t_mrh_camera = transformation_file.getNode("t_mtx").mat()
    #     transformation_file.release()
    #     T_mrh_camera[:3, :
    #                  3], T_mrh_camera[:3,
    #                                   3] = R_mrh_camera, t_mrh_camera.reshape(
    #                                       -1)
    #     T_camera_mrh = np.linalg.inv(T_mrh_camera)
    #     T_camera_ego = T_mrh_ego @ T_camera_mrh

    #     # Get ground truth cone arrays in mrh frame
    #     gt_cone_arrays = []

    #     gt_cone_array_indices = []
    #     gt_cone_array_folder = os.path.join(label_paths_file,
    #                                         camera + "_cones_corrected")

    #     for (_, _, current_files) in os.walk(gt_cone_array_folder):
    #         gt_cone_array_files = [
    #             os.path.join(gt_cone_array_folder, current_file)
    #             for current_file in current_files
    #         ]

    #         for current_file in current_files:
    #             filename, _ = os.path.splitext(current_file)
    #             gt_cone_array_indices.append(int(filename))

    #         break

    #     for idx, gt_cone_array_file in enumerate(gt_cone_array_files):
    #         gt_cone_array = np.fromfile(gt_cone_array_file).reshape(-1, 4)
    #         cone_types = gt_cone_array[:, 0]
    #         cone_positions = gt_cone_array[:, 1:]
    #         h_cone_positions = np.hstack(
    #             (cone_positions, np.ones((cone_positions.shape[0], 1))))
    #         h_cone_positions = T_camera_ego @ h_cone_positions.T
    #         gt_cone_array = np.hstack(
    #             (cone_types.reshape(-1, 1), h_cone_positions.T[:, :3]))
    #         gt_cone_arrays_dict[camera][
    #             gt_cone_array_indices[idx]] = gt_cone_array

    #     indices.extend(list(gt_cone_arrays_dict[camera].keys()))

    # # Get all indices for which we have gt cone arrays
    # u, c = np.unique(np.array(indices), return_counts=True)
    # indices = u[c > 0]

    # # Get lidar pipeline cone arrays
    # lidar_cone_arrays = []

    # lidar_cone_array_folder = os.path.join(label_paths_file,
    #                                        "lidar_cone_arrays_filtered")
    # lidar_cone_array_files = []
    # for (_, _, current_files) in os.walk(lidar_cone_array_folder):
    #     lidar_cone_array_files.extend(current_files)
    # lidar_cone_array_files.remove("timestamps.txt")
    # lidar_cone_array_files.sort()
    # for lidar_cone_array_file in lidar_cone_array_files:
    #     lidar_cone_array_file = os.path.join(lidar_cone_array_folder,
    #                                          lidar_cone_array_file)
    #     lidar_cone_array = np.fromfile(lidar_cone_array_file).reshape(-1, 3)
    #     lidar_cone_arrays.append(lidar_cone_array)

    # # Filter comparable data
    # cone_arrays_dict = {"lidar": [], "gt": []}
    # gt_indices = []

    # for idx in indices:
    #     lidar_cone_array = lidar_cone_arrays[idx]
    #     if lidar_cone_array.shape[0] != 0:
    #         cone_arrays_dict["lidar"].append(lidar_cone_array)
    #         gt_indices.append(idx)

    # for idx in gt_indices:
    #     if idx in gt_cone_arrays_dict["forward"]:
    #         cone_arrays_dict["gt"].append(gt_cone_arrays_dict["forward"][idx])
    #     elif idx in gt_cone_arrays_dict["right"]:
    #         cone_arrays_dict["gt"].append(gt_cone_arrays_dict["right"][idx])
    #     elif idx in gt_cone_arrays_dict["left"]:
    #         cone_arrays_dict["gt"].append(gt_cone_arrays_dict["left"][idx])
    #     else:
    #         print("There exists no ground truth cone array with this index!")
    #         sys.exit()

    return cone_arrays_dict


def visualize_cone_arrays(cfg):
    cone_arrays_dict = match_cone_arrays(cfg)
    num_cone_arrays = len(cone_arrays_dict["lidar"])

    for cone_array_idx in range(num_cone_arrays):
        pcd_lidar = o3d.geometry.PointCloud()
        pcd_lidar.points = o3d.utility.Vector3dVector(
            cone_arrays_dict["lidar"][cone_array_idx])
        pcd_lidar.paint_uniform_color([1.0, 0.1, 0.1])

        pcd_gt = o3d.geometry.PointCloud()
        # Project the gt cones onto the xy plane
        gt_cone_array_projected = np.hstack(
            (cone_arrays_dict["gt"][cone_array_idx][:, 1:3],
             np.zeros((cone_arrays_dict["gt"][cone_array_idx].shape[0], 1))))
        # Artificially make a point to have a z!=0 to counter the open3d visualization bug
        gt_cone_array_projected[-1, -1] = 1
        pcd_gt.points = o3d.utility.Vector3dVector(gt_cone_array_projected)
        pcd_gt.paint_uniform_color([0.1, 0.9, 0.1])

        # Coordinate Frame
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0])

        o3d.visualization.draw_geometries([pcd_lidar, pcd_gt, mesh_frame])

    print("Done")


def calculate_metrics(cfg):
    MAX_DIST_PREDICTED_TO_GT_CONE = 0.5

    cone_arrays_dict = match_cone_arrays(cfg)
    num_cone_arrays = len(cone_arrays_dict["lidar"])

    depth_metric = np.zeros((2, cfg.max_distance // cfg.interval_length))

    for idx in range(num_cone_arrays):
        gt_cone_array = o3d.geometry.PointCloud()
        gt_cone_array.points = o3d.utility.Vector3dVector(
            cone_arrays_dict["gt"][idx][:, 1:])
        gt_cone_array_tree = o3d.geometry.KDTreeFlann(gt_cone_array)

        for lidar_cone in cone_arrays_dict["lidar"][idx]:

            [k, idx_nn,
             _] = gt_cone_array_tree.search_knn_vector_3d(lidar_cone, 1)
            [k_, idx_ball, _] = gt_cone_array_tree.search_radius_vector_3d(
                lidar_cone, MAX_DIST_PREDICTED_TO_GT_CONE)

            if idx_nn[0] in idx_ball:
                gt_cone = cone_arrays_dict["gt"][idx][idx_nn[0], 1:]
                dist_gt_origin = np.linalg.norm(gt_cone)
                if dist_gt_origin >= cfg.max_distance:
                    continue
                dist_gt_origin_array_idx = int(dist_gt_origin /
                                               cfg.interval_length)
                dist_gt_lidar = np.linalg.norm(gt_cone - lidar_cone)

                depth_metric[0, dist_gt_origin_array_idx] += 1
                depth_metric[1, dist_gt_origin_array_idx] += dist_gt_lidar

    for idx in range(depth_metric.shape[1]):
        if depth_metric[0, idx] != 0:
            depth_metric[1, idx] = depth_metric[1, idx] / depth_metric[0, idx]

    plt.style.use("fivethirtyeight")
    dist_ranges = np.arange(0, cfg.max_distance, cfg.interval_length).tolist()
    x_dist_ranges = [
        str(dist_ranges[i]) + " - " + str(dist_ranges[i + 1])
        for i in range(len(dist_ranges) - 1)
    ]
    y_num_predictions = depth_metric[0, :]
    y_average_dist_error = depth_metric[1, :]

    plt.bar(x_dist_ranges,
            y_num_predictions,
            color='#444444',
            label='Number of Predictions')
    plt.bar(x_dist_ranges,
            y_average_dist_error,
            color='#777777',
            label='Average Distance Error')

    plt.legend()
    plt.show()

    print("Done")


def main():
    cfg = command_line_parser()
    check_cfg(cfg)

    if cfg.mode == "vis":
        visualize_cone_arrays(cfg)
    elif cfg.mode == "metrics":
        calculate_metrics(cfg)
    else:
        print("Moop")


if __name__ == "__main__":
    main()
