import argparse
import sys
import os
import re
import yaml
import numpy as np
import pickle5 as pickle
import cv2
import pathlib
import open3d as o3d
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.spatial import cKDTree


def command_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-b',
        '--data',
        default='/media/benjin/Samsung_T5/AMZ/sensor_fusion_data',
        type=str,
        help='Specify local path of the sensor_fusion_data folder')

    parser.add_argument(
        '-l',
        '--label_paths_file',
        default=
        '/home/benjin/Development/git/sensor_fusion_2020/data/full/test_sf.txt',
        type=str,
        help='Specify which data split to calculate metrics on')

    parser.add_argument('--cached_data',
                        action='store_true',
                        help='Use the cached data instead')

    parser.add_argument('-m',
                        '--mode',
                        default='vis',
                        type=str,
                        choices=['vis', 'save_vis', 'metrics'],
                        help='Specify the mode of the program')

    parser.add_argument(
        '-md',
        '--max_distance',
        default=60,
        type=int,
        help=
        'Maximal expected prediction distance. Every prediction above will be discarded.'
    )

    parser.add_argument('-i',
                        '--interval_length',
                        default=3,
                        type=int,
                        help='Interval length for grouping predicted cones.')

    cfg = parser.parse_args()

    return cfg


def check_cfg(cfg):
    if cfg.max_distance % cfg.interval_length != 0:
        print("Command line arguments invalid!")
        sys.exit()


def extract_camera_from_path(gt_cone_array_path):
    cameras = ["left", "right", "forward"]
    for camera in cameras:
        if gt_cone_array_path.find(camera) != -1:
            return camera

    print("No camera could be extracted from path!")
    sys.exit()


def match_cone_arrays(cfg):
    """
    Read in the lidar pipeline predictions and read in the ground truth cone
    arrays. 
    """
    def duplicate_gt(gt_cone_array_path, gt_cone_array_paths):
        reg_str = re.compile(
            gt_cone_array_path.replace('forward',
                                       '.+').replace('left', '.+').replace(
                                           'right', '.+'))
        num_duplicates = sum(
            [bool(reg_str.match(path)) for path in gt_cone_array_paths])
        return num_duplicates > 0

    def get_T_mrh_to_ego(static_transformations_folder):
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
        p = pathlib.Path(test_day)
        return p.parts[1]

    def load_rosparam_mat(yaml_dict, param_name):
        """Given a dict intended for ROS, convert the list to a matrix for a given key"""
        rows = yaml_dict[param_name]['rows']
        cols = yaml_dict[param_name]['cols']
        data = yaml_dict[param_name]['data']

        return np.asarray(data).reshape((rows, cols))

    def load_camera_calib(camera_fn):

        if not os.path.exists(camera_fn):
            print("Calibration file does not exist!")
            exit()

        calib = {}
        with open(camera_fn) as camera_f:
            try:
                camera_data = yaml.load(camera_f, Loader=yaml.CLoader)
            except AttributeError:
                camera_data = yaml.load(camera_f, Loader=yaml.Loader)

        calib['camera_matrix'] = load_rosparam_mat(camera_data,
                                                   'camera_matrix')
        calib['distortion_coefficients'] = load_rosparam_mat(
            camera_data, 'distortion_coefficients')
        calib['image_width'] = camera_data['image_width']
        calib['image_height'] = camera_data['image_height']
        calib['projection_matrix'] = load_rosparam_mat(camera_data,
                                                       'projection_matrix')

        return calib

    def get_gt_cone_array(gt_cone_array_path):
        """
        Parameter: 
        - gt_cone_array_path: path of the gt cone array file
        Return:
        - the gt cone array in the egomotion frame
        """

        # Load necessary transformations
        test_day = extract_test_day_from_path(gt_cone_array_path, cfg.data)

        T_mrh_ego = get_T_mrh_to_ego(
            os.path.join(cfg.data, test_day, "static_transformations"))

        camera = extract_camera_from_path(gt_cone_array_path)

        T_camera_mrh = get_T_camera_to_mrh(
            os.path.join(cfg.data, test_day, "static_transformations"), camera)

        T_camera_ego = T_mrh_ego @ T_camera_mrh

        # Load gt cone array and transform to egomotion
        gt_cone_array = np.fromfile(gt_cone_array_path).reshape(-1, 4)
        cone_types = gt_cone_array[:, 0]
        cone_positions = gt_cone_array[:, 1:]

        h_cone_positions = np.hstack(
            (cone_positions, np.ones((cone_positions.shape[0], 1))))
        h_cone_positions = T_camera_ego @ h_cone_positions.T
        gt_cone_array = np.hstack(
            (cone_types.reshape(-1, 1), h_cone_positions.T[:, :3]))

        return gt_cone_array

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

    def get_lidar_cone_array(lidar_cone_array_path):
        return np.fromfile(lidar_cone_array_path).reshape(-1, 3)

    #####################################################

    cone_arrays_dict = {}

    # Read in file paths of the labels of the data split
    with open(cfg.label_paths_file, "r") as f:
        orig_label_paths = f.readlines()

    # Find index at which to split the label file paths
    base_folder_name = os.path.basename(cfg.data)
    index = orig_label_paths[0].find(base_folder_name)
    index += len(base_folder_name) + 1

    pbar = tqdm(total=len(orig_label_paths), desc="Extracting cone arrays")

    gt_cone_array_paths = []
    lidar_cone_array_paths = []
    counter_1, counter_2, counter_3 = 0, 0, 0
    for label_path in orig_label_paths:
        label_path = os.path.join(cfg.data, label_path[index:].rstrip())
        pbar.update(1)
        if os.path.exists(label_path):
            gt_cone_array_path = label_path.replace('labels',
                                                    'cones_corrected').replace(
                                                        'txt', 'bin')

            if os.path.exists(gt_cone_array_path) and not duplicate_gt(
                    gt_cone_array_path, gt_cone_array_paths):
                # - [ ]  Filter out duplicates, meaning gt cone arrays that belong to the same timestamp
                # - [ ]  For each gt_cone_array, find the corresponding lidar file path, skip if not found
                lidar_cone_array_path = get_lidar_cone_array_path(
                    gt_cone_array_path)
                if lidar_cone_array_path is not None:

                    # Find index at which to split the lidar cone array path
                    KEYWORD = "data"
                    index = lidar_cone_array_path.find(KEYWORD)
                    index += len(KEYWORD) + 1

                    sample = {}
                    sample["gt"] = get_gt_cone_array(gt_cone_array_path)
                    sample["lidar"] = get_lidar_cone_array(
                        lidar_cone_array_path)
                    cone_arrays_dict[lidar_cone_array_path[index:]] = sample

                else:
                    counter_3 += 1
            else:
                counter_2 += 1
        else:
            counter_1 += 1

    pbar.close()

    if counter_1:
        print(
            "{} specified label files don't exist on the local system. Data bases out of sync?"
            .format(counter_1))
    if counter_2:
        print(
            "{} specified gt cone array files are duplicates or don't exist on the local system."
            .format(counter_2))
    if counter_3:
        print(
            "{} specified lidar cone array files don't exist on the local system. Have they been extracted?"
            .format(counter_3))

    # Cache cone arrays dict
    with open("cone_arrays_dict.pkl", "wb") as f:
        pickle.dump(cone_arrays_dict, f)

    return cone_arrays_dict


def visualize_cone_arrays(cfg):
    """
    Loading in the predicted 3D cone arrays from the lidar pipeline and the
    ground truth cone arrays for visualization in BEV perspective.
    """

    # Loading of ground truth and predicted cone arrays + bounding boxes
    if cfg.cached_data:
        try:
            with open("cone_arrays_dict.pkl", "rb") as f:
                cone_arrays_dict = pickle.load(f)
        except:
            print("No cache file")
            cone_arrays_dict = match_cone_arrays(cfg)
    else:
        cone_arrays_dict = match_cone_arrays(cfg)

    # Generating visualizations
    plt.style.use('seaborn')

    for index, (key, data_item) in enumerate(cone_arrays_dict.items()):

        gt_cone_array = data_item["gt"]
        lidar_cone_array = data_item["lidar"]

        # Create scatter plot, neglect z axis
        fig1, (ax1_1) = plt.subplots(nrows=1, ncols=1)

        # Manually place ticks
        SPACING = 10.0
        xmin = min([
            np.min(gt_cone_array[:, 1]),
            np.min(lidar_cone_array[:, 0]),
        ])
        xmax = max([
            np.max(gt_cone_array[:, 1]),
            np.max(lidar_cone_array[:, 0]),
        ])
        ymin = min([
            np.min(gt_cone_array[:, 2]),
            np.min(lidar_cone_array[:, 1]),
        ])
        ymax = max([
            np.max(gt_cone_array[:, 2]),
            np.max(lidar_cone_array[:, 1]),
        ])
        xticks = np.arange(np.floor(xmin / SPACING),
                           np.ceil(xmax / SPACING) + 1) * SPACING
        yticks = np.arange(np.floor(ymin / SPACING),
                           np.ceil(ymax / SPACING) + 1) * SPACING

        fig1.set_figheight(len(yticks))
        fig1.set_figwidth(len(xticks))

        # Plot ground truth cones
        gt_c = np.array([
            np.array([0.0, 0.0, 0.0, 1.0])
            if x == 0 else np.array([1.0, 1.0, 1.0, 1.0])
            for x in gt_cone_array[:, 0]
        ])
        ax1_1.scatter(gt_cone_array[:, 1], gt_cone_array[:, 2], c=gt_c)

        # Plot lidar cone array
        ax1_1.scatter(lidar_cone_array[:, 0],
                      lidar_cone_array[:, 1],
                      c="Gray",
                      edgecolor="Black")

        # Find index at which to split the lidar cone array path
        KEYWORD = "data"
        i = key.find(KEYWORD)
        i += len(KEYWORD) + 1

        ax1_1.set_title("BEV: " +
                        key[i:].replace("lidar_cone_arrays_filtered", "..."))
        ax1_1.set_xlabel("x/m")
        ax1_1.set_ylabel("y/m")
        ax1_1.set_xticks(xticks)
        ax1_1.set_yticks(yticks)

        plt.tight_layout()
        if cfg.mode == "save_vis":
            plt.savefig("bev_" + str(index).zfill(4))
        if cfg.mode == "vis":
            plt.show()

    print("Done")


def calculate_metrics(cfg):
    """
    Loading in the predicted 3D cone arrays, bounding boxes + depth and the
    ground truth cone arrays for calculating 3D detection errors.
    """

    # Loading of ground truth and predicted cone arrays + bounding boxes
    if cfg.cached_data:
        try:
            with open("cone_arrays_dict.pkl", "rb") as f:
                cone_arrays_dict = pickle.load(f)
        except:
            print("No cache file")
            cone_arrays_dict = match_cone_arrays(cfg)
    else:
        cone_arrays_dict = match_cone_arrays(cfg)

    # Setting the maximal correspondence distance
    MAX_DIST_PREDICTED_TO_GT_CONE = 2

    num_bins = cfg.max_distance // cfg.interval_length

    pos_error = [[] for _ in range(num_bins)]
    std_pos_error = [0.0] * num_bins
    avg_pos_error = [0.0] * num_bins
    num_predictions = [0] * num_bins

    # Iterate over the data items of every timestamp
    # For every predicted cond find its corresponding ground truth cone
    # and store the distance error
    for key, data_item in cone_arrays_dict.items():
        gt_cone_array = data_item["gt"]
        lidar_cone_array = data_item["lidar"]

        gt_cone_tree = cKDTree(gt_cone_array[:, 1:])

        for lidar_cone in lidar_cone_array:
            dd, ii = gt_cone_tree.query(lidar_cone, distance_upper_bound=2)
            if ii != gt_cone_tree.n:
                gt_cone = gt_cone_array[ii]
                dist_gt_origin = np.linalg.norm(gt_cone)

                if dist_gt_origin >= cfg.max_distance:
                    continue

                dist_gt_origin_array_idx = int(dist_gt_origin /
                                               cfg.interval_length)
                pos_error[dist_gt_origin_array_idx].append(dd)

    # Calculate metrics
    for idx in range(num_bins):
        if pos_error[idx]:
            avg_pos_error[idx] = sum(pos_error[idx]) / len(pos_error[idx])
            std_pos_error[idx] = float(np.std(np.array(pos_error[idx])))
            num_predictions[idx] = len(pos_error[idx])

    # Visualize metrics
    plt.style.use("seaborn")
    dist_ranges = np.linspace(0, cfg.max_distance,
                              int(cfg.max_distance / cfg.interval_length +
                                  1)).astype(np.int).tolist()
    x_dist_ranges = [
        str(dist_ranges[i]) + "-" + str(dist_ranges[i + 1])
        for i in range(len(dist_ranges) - 1)
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)

    fig.set_figheight(5)
    fig.set_figwidth(10)

    ax1.bar(x_dist_ranges,
            num_predictions,
            color='#444444',
            label='Number of Predictions')
    # ax1.set_ylabel("Number of Predictions")
    ax1.set_title("3D Cone Prediction Evaluation")
    ax1.legend()
    ax2.bar(x_dist_ranges,
            avg_pos_error,
            color='#777777',
            label='Average Distance Error')
    # ax2.set_ylabel("Average Distance Error")
    ax2.legend()
    ax3.bar(x_dist_ranges,
            std_pos_error,
            color='#aaaaaa',
            label='Std of Distance Errors')
    ax3.set_xlabel("Distance Range")
    # ax3.set_ylabel("Std of Distance Error")
    ax3.legend()
    plt.tight_layout()
    plt.show()

    print("Done")


def main():
    cfg = command_line_parser()
    check_cfg(cfg)

    if cfg.mode == "vis" or cfg.mode == "save_vis":
        visualize_cone_arrays(cfg)
    elif cfg.mode == "metrics":
        calculate_metrics(cfg)
    else:
        print("Moop. Mode not implemented.")


if __name__ == "__main__":
    main()