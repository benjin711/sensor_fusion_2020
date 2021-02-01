import argparse
import sys
import os
import re
import yaml
import numpy as np
import pickle5 as pickle
import cv2
import pathlib
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.spatial import cKDTree
from matplotlib.lines import Line2D
import matplotlib as mpl


def command_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--pred',
        default=
        '/media/benjin/Samsung_T5/AMZ/sensor_fusion_inference/inference.pkl',
        type=str,
        help='Specify path of the pickle file that contains the predictions')

    parser.add_argument(
        '--data',
        default='/media/benjin/Samsung_T5/AMZ/sensor_fusion_data',
        type=str,
        help='Specify the base folder containing all the data')

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

    parser.add_argument('--bb_depth',
                        action='store_true',
                        help='Calculate depth based on bounding box size')

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
    Read in the predictions provided by test.py file and read in the ground truth cone
    arrays. Calculate the 3D cones given the predictions and transform the resulting 
    cone array in the egomotion frame. 
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

    def get_sf_cone_array(data_folder, key, predictions, bb_depth=False):
        """
        Parameter: 
        - data_folder: path of the data folder (sensor_fusion_data)
        - key: key to the cone array prediction
        - predictions: dictionary containing all predicted cone arrays
        Return:
        - the predicted sensor fusion cone array in the egomotion frame
        """

        # Check that we have predictions for left and right camera images
        camera_key_dict = {}
        which_camera = extract_camera_from_path(key)
        for camera in ["left", "right"]:
            key_ = key.replace(which_camera, camera)

            if key_ not in predictions:
                return None, None, False
            camera_key_dict[camera] = key_

        # Load bounding box arrays and calculate the corresponding cone arrays
        cone_arrays = {}
        bb_arrays = {}

        for (camera, key_) in camera_key_dict.items():
            cxywhdc_array = predictions[key_]

            # Back project predictions to 3D using 2D cone tip position and depth
            img_path = os.path.join(cfg.data, key_)
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            cone_type = cxywhdc_array[:, 0]
            xywhn_depth_array = cxywhdc_array[:, 1:6]
            conf = cxywhdc_array[:, 6]

            xyd_array = np.array([
                xywhn_depth_array[:, 0] * w,
                xywhn_depth_array[:, 1] * h - 0.5 * xywhn_depth_array[:, 3],
                xywhn_depth_array[:, 4]
            ]).T

            # Convert xywhn_depth to xyxyd
            xywh_array = np.array([
                xyd_array[:, 0], xyd_array[:, 1], xywhn_depth_array[:, 2] * w,
                xywhn_depth_array[:, 3] * h
            ]).T
            xyxydc_array = np.array([
                xyd_array[:, 0] - xywh_array[:, 2] / 2,
                xyd_array[:, 1] - xywh_array[:, 3] / 2,
                xyd_array[:, 0] + xywh_array[:, 2] / 2,
                xyd_array[:, 1] + xywh_array[:, 3] / 2, xyd_array[:,
                                                                  2], cone_type
            ]).T
            bb_arrays[camera] = xyxydc_array

            # Load camera intrinsics
            cam_calib_file = os.path.join(os.path.dirname(img_path),
                                          "../../../static_transformations",
                                          camera + ".yaml")
            K = load_camera_calib(cam_calib_file)["camera_matrix"]

            # Recalculate depth based on bb size instead of network predicted depth
            if bb_depth:
                CONE_HEIGHT = 0.325
                Z = K[1, 1] / xywh_array[:, 3] * CONE_HEIGHT
                X = np.abs(xywh_array[:, 0] - 0.5 * w) / K[0, 0] * Z
                Y = np.abs(xywh_array[:, 1] - 0.5 * h) / K[1, 1] * Z
                d = np.sqrt(np.square(X) + np.square(Y) + np.square(Z))
                xyd_array[:, 2] = d

            # Backproject to 3D
            h_xy = np.hstack((xyd_array[:, :2], np.ones(
                (xyd_array.shape[0], 1))))
            d = xyd_array[:, 2]
            c_xyz_s = np.linalg.inv(K) @ h_xy.T
            scalars = d / np.linalg.norm(c_xyz_s.T, axis=1)
            c_xyz = (c_xyz_s * scalars).T

            # Transform to egomotion coordinate frame
            test_day = extract_test_day_from_path(gt_cone_array_path, cfg.data)

            T_mrh_ego = get_T_mrh_to_ego(
                os.path.join(cfg.data, test_day, "static_transformations"))

            T_camera_mrh = get_T_camera_to_mrh(
                os.path.join(cfg.data, test_day, "static_transformations"),
                camera)

            T_camera_ego = T_mrh_ego @ T_camera_mrh

            h_c_xyz = np.hstack((c_xyz, np.ones((c_xyz.shape[0], 1))))
            e_xyz = (T_camera_ego @ h_c_xyz.T).T[:, :3]

            cone_arrays[camera] = np.hstack(
                (e_xyz, cone_type[:, None], conf[:, None]))

        return cone_arrays, bb_arrays, True

    #####################################################

    cone_arrays_dict = {}

    with open(cfg.pred, "rb") as f:
        predictions = pickle.load(f)

    img_paths = list(predictions.keys())

    pbar = tqdm(total=len(img_paths), desc="Extracting cone arrays")
    gt_cone_array_paths = []
    counter = 0
    for img_path in img_paths:
        local_img_path = os.path.join(cfg.data, img_path.rstrip())
        if os.path.exists(local_img_path):
            gt_cone_array_path = local_img_path.replace(
                'camera_filtered', 'cones_corrected').replace('png', 'bin')

            # Filter out duplicates, meaning gt cone arrays that belong to the same timestamp, i.e.
            # right/left/forward_cones_corrected/00000123.bin files contain the same gt cones only in different
            # coordinate systems
            if os.path.exists(gt_cone_array_path) and not duplicate_gt(
                    gt_cone_array_path, gt_cone_array_paths):

                sample = {}
                sample["gt_cone_array"] = get_gt_cone_array(gt_cone_array_path)
                sample["sf_cone_array"], sample[
                    "sf_bb_array"], success = get_sf_cone_array(
                        cfg.data, img_path, predictions, cfg.bb_depth)

                if success:
                    cone_arrays_dict[img_path] = sample

            pbar.update(1)

        else:
            counter += 1

    if counter:
        print(
            "{} specified img files don't exist on the local system. Data bases out of sync?"
            .format(counter))

    # Cache cone arrays dict
    with open("sf_cone_arrays_dict.pkl", "wb") as f:
        pickle.dump(cone_arrays_dict, f)

    return cone_arrays_dict


def visualize_cone_arrays(cfg):
    """
    Loading in the predicted 3D cone arrays, bounding boxes + depth and the
    ground truth cone arrays for visualization of the cones in BEV and the bb 
    in the images.
    """

    # Loading of ground truth and predicted cone arrays + bounding boxes
    if cfg.cached_data:
        try:
            with open("sf_cone_arrays_dict.pkl", "rb") as f:
                cone_arrays_dict = pickle.load(f)
        except:
            print("No cache file")
            cone_arrays_dict = match_cone_arrays(cfg)
    else:
        cone_arrays_dict = match_cone_arrays(cfg)

    # Generating visualizations
    plt.style.use('seaborn')

    for index, (key, data_item) in enumerate(cone_arrays_dict.items()):
        which_camera = extract_camera_from_path(key)
        img_path = {}
        img = {}
        keys = {}
        for camera in ["left", "right"]:
            keys[camera] = key.replace(which_camera, camera)
            img_path[camera] = os.path.join(cfg.data,
                                            key.replace(which_camera, camera))
            img[camera] = cv2.imread(img_path[camera])

        gt_cone_array = data_item["gt_cone_array"]
        sf_cone_array_left = data_item["sf_cone_array"]["left"]
        sf_cone_array_right = data_item["sf_cone_array"]["right"]
        sf_bb_array_left = data_item["sf_bb_array"]["left"]
        sf_bb_array_right = data_item["sf_bb_array"]["right"]

        # Create scatter plot, neglect z axis
        fig1, (ax1_1) = plt.subplots(nrows=1, ncols=1)

        # Manually place ticks
        SPACING = 10.0
        xmin = min([
            np.min(gt_cone_array[:, 1]),
            np.min(sf_cone_array_left[:, 0]),
            np.min(sf_cone_array_right[:, 0])
        ])
        xmax = max([
            np.max(gt_cone_array[:, 1]),
            np.max(sf_cone_array_left[:, 0]),
            np.max(sf_cone_array_right[:, 0])
        ])
        ymin = min([
            np.min(gt_cone_array[:, 2]),
            np.min(sf_cone_array_left[:, 1]),
            np.min(sf_cone_array_right[:, 1])
        ])
        ymax = max([
            np.max(gt_cone_array[:, 2]),
            np.max(sf_cone_array_left[:, 1]),
            np.max(sf_cone_array_right[:, 1])
        ])
        xticks = np.arange(
            np.floor(xmin / SPACING),
            np.ceil(xmax / SPACING) + 1 +
            1) * SPACING  # Second + 1 is to have extra space for the legend
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

        # Plot sf cone array left
        sf_c_left = np.array([
            np.array([0.0, 0.0, 1.0, 1.0])
            if x == 0 else np.array([1.0, 1.0, 0.0, 1.0])
            for x in sf_cone_array_left[:, 3]
        ])
        sf_c_left[:, 3] = sf_cone_array_left[:, 4]
        ax1_1.scatter(sf_cone_array_left[:, 0],
                      sf_cone_array_left[:, 1],
                      c=sf_c_left,
                      edgecolor="Black")

        # Plot sf cone array right
        sf_c_right = np.array([
            np.array([0.0, 0.0, 1.0, 1.0])
            if x == 0 else np.array([1.0, 1.0, 0.0, 1.0])
            for x in sf_cone_array_right[:, 3]
        ])
        sf_c_right[:, 3] = sf_cone_array_right[:, 4]
        ax1_1.scatter(sf_cone_array_right[:, 0],
                      sf_cone_array_right[:, 1],
                      c=sf_c_right,
                      edgecolor="Black")

        # Find index at which to split the lidar cone array path
        KEYWORD = "data"
        i = key.find(KEYWORD)
        i += len(KEYWORD) + 1

        ax1_1.set_title("BEV: " +
                        key[i:].replace("right_camera_filtered", "...").
                        replace("left_camera_filtered", "..."))
        ax1_1.set_xlabel("x/m")
        ax1_1.set_ylabel("y/m")
        ax1_1.set_xticks(xticks)
        ax1_1.set_yticks(yticks)

        # Create legend manually
        colors = ['black', 'blue', 'yellow', 'white']
        circles = [
            Line2D([0], [0],
                   color=mpl.rcParams["axes.facecolor"],
                   marker='o',
                   markerfacecolor=c,
                   markeredgecolor=c) for c in colors
        ]
        labels = [
            'GT Blue Cone',
            'YOLO3D Blue Cone',
            'YOLO3D Yellow Cone',
            'GT Yellow Cone',
        ]
        ax1_1.legend(circles, labels)

        plt.tight_layout()
        if cfg.mode == "save_vis":
            plt.savefig("bev_" + str(index).zfill(4))
        if cfg.mode == "vis":
            plt.show()

        # Create image plot
        fig2, (ax2_1, ax2_2) = plt.subplots(nrows=1, ncols=2)
        fig2.set_figheight(3)
        fig2.set_figwidth(16)

        # Plot left image with bounding boxes
        img_l = cv2.cvtColor(img["left"], cv2.COLOR_BGR2RGB)
        img_l = deepcopy(img_l)
        for xyxydc in sf_bb_array_left:
            color = (255, 255, 0) if xyxydc[5] else (0, 0, 255)
            x1, y1 = tuple(xyxydc[:2].astype(np.int))
            x2, y2 = tuple(xyxydc[2:4].astype(np.int))
            cv2.rectangle(img_l, (int(x1), int(y1)), (int(x2), int(y2)), color,
                          2)
        ax2_1.imshow(img_l)
        ax2_1.set_title(keys["left"][keys["left"].find("data") + len("data") +
                                     1:])
        ax2_1.set_xticks([])
        ax2_1.set_yticks([])

        # Plot right image with bounding boxes
        img_r = cv2.cvtColor(img["right"], cv2.COLOR_BGR2RGB)
        img_r = deepcopy(img_r)
        for xyxydc in sf_bb_array_right:
            color = (255, 255, 0) if xyxydc[5] else (0, 0, 255)
            x1, y1 = tuple(xyxydc[:2].astype(np.int))
            x2, y2 = tuple(xyxydc[2:4].astype(np.int))
            cv2.rectangle(img_r, (int(x1), int(y1)), (int(x2), int(y2)), color,
                          2)
        ax2_2.imshow(img_r)
        ax2_2.set_title(keys["right"][keys["right"].find("data") +
                                      len("data") + 1:])
        ax2_2.set_xticks([])
        ax2_2.set_yticks([])
        plt.tight_layout(w_pad=2)
        if cfg.mode == "save_vis":
            plt.savefig("imgs_" + str(index).zfill(4))
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
            with open("sf_cone_arrays_dict.pkl", "rb") as f:
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
        gt_cone_array = data_item["gt_cone_array"]
        sf_cone_array_left = data_item["sf_cone_array"]["left"]
        sf_cone_array_right = data_item["sf_cone_array"]["right"]
        sf_cone_array = np.vstack((sf_cone_array_left, sf_cone_array_right))

        gt_cone_tree = cKDTree(gt_cone_array[:, 1:])

        for sf_cone in sf_cone_array[:, :3]:
            dd, ii = gt_cone_tree.query(sf_cone, distance_upper_bound=2)
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
    ax1.set_title("3D Cone Prediction Evaluation")
    ax1.legend()
    ax2.bar(x_dist_ranges,
            avg_pos_error,
            color='#777777',
            label='Average Distance Error')
    ax2.legend()
    ax3.bar(x_dist_ranges,
            std_pos_error,
            color='#aaaaaa',
            label='Std of Distance Errors')
    ax3.set_xlabel("Distance Range")
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