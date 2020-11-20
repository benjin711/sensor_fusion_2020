import argparse
import os
import numpy as np
import sys
import cv2
import open3d

# Would be good to know which data folders carter is using as test data atm


def command_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-d',
        '--data_folders',
        nargs="+",
        default=[
            '/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-07-05_tuggen/data/autocross_2020-07-05-11-58-07',
            # '/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-07-05_tuggen/data/autocross_2020-07-05-12-35-31',
            # '/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-07-05_tuggen/data/autocross_2020-07-05-13-57-26',
            # '/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-07-08_duebendorf/data/autocross_2020-07-08-09-53-46',
            # '/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-07-12_tuggen/data/autocross_2020-07-12-09-54-31',
            # '/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-07-12_tuggen/data/autocross_2020-07-12-10-00-35',
            # '/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-07-19_tuggen/data/autocross_2020-07-19-15-18-28',
            # '/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-08-01_tuggen/data/autocross_2020-08-01-14-45-23',
            # '/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-08-01_tuggen/data/autocross_2020-08-01-18-30-54',
            # '/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-08-02_tuggen/data/autocross_2020-08-02-12-01-19',
            # '/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-08-02_tuggen/data/autocross_2020-08-02-09-26-48',
            # '/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-08-12_duebendorf/data/autocross_2020-08-12-13-15-14',
            # '/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-08-23_tuggen/data/autocross_2020-08-23-08-10-36',
            # '/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-08-08_alpnach/data/autocross_2020-08-08-13-46-33'
        ],
        type=str,
        help='Specify which data to use to calculate metrics')

    parser.add_argument('-m',
                        '--mode',
                        default='vis',
                        type=str,
                        choices=['vis, metrics'],
                        help='Specify what the program should do')

    cfg = parser.parse_args()

    return cfg


def match_cone_arrays(cfg):
    # Get ground truth cone arrays in mrh_lidar frame
    # Use the GT cone arrays from the left and right camera frames
    gt_cone_arrays_dict = {"left": {}, "right": {}}
    indices = []

    # Pathing
    data_folder = cfg.data_folders[0]
    static_transformations_folder = os.path.join(data_folder, "..", "..",
                                                 "static_transformations")

    for camera in gt_cone_arrays_dict.keys():
        # Get static transformation
        T = np.eye(4)

        T_file = os.path.join(static_transformations_folder,
                              "extrinsics_mrh_" + camera + ".yaml")
        transformation_file = cv2.FileStorage(T_file, cv2.FILE_STORAGE_READ)
        R = transformation_file.getNode("R_mtx").mat()
        t = transformation_file.getNode("t_mtx").mat()
        transformation_file.release()
        T[:3, :3], T[:3, 3] = R, t.reshape(-1)

        # Get ground truth cone arrays in mrh frame
        gt_cone_arrays = []

        gt_cone_array_indices = []
        gt_cone_array_folder = os.path.join(data_folder,
                                            camera + "_cones_corrected")

        for (_, _, current_files) in os.walk(gt_cone_array_folder):
            gt_cone_array_files = [
                os.path.join(gt_cone_array_folder, current_file)
                for current_file in current_files
            ]

            for current_file in current_files:
                filename, _ = os.path.splitext(current_file)
                gt_cone_array_indices.append(int(filename))

            break

        for idx, gt_cone_array_file in enumerate(gt_cone_array_files):
            gt_cone_array = np.fromfile(gt_cone_array_file).reshape(-1, 4)
            cone_types = gt_cone_array[:, 0]
            cone_positions = gt_cone_array[:, 1:]
            h_cone_positions = np.hstack(
                (cone_positions, np.ones((cone_positions.shape[0], 1))))
            h_cone_positions = T @ h_cone_positions.T
            gt_cone_array = np.hstack(
                (cone_types.reshape(-1, 1), h_cone_positions.T))[:, :4]
            gt_cone_arrays_dict[camera][
                gt_cone_array_indices[idx]] = gt_cone_array

        indices.extend(list(gt_cone_arrays_dict[camera].keys()))

    # Filter indices when there were gt cones for the left and right camera
    u, c = np.unique(np.array(indices), return_counts=True)
    indices = u[c > 1]

    # Get lidar pipeline cone arrays
    lidar_cone_arrays = []

    lidar_cone_array_folder = os.path.join(data_folder, "lidar_cone_arrays")
    lidar_cone_array_files = []
    for (_, _, current_files) in os.walk(lidar_cone_array_folder):
        lidar_cone_array_files.extend(current_files)
    lidar_cone_array_files.remove("timestamps.txt")
    lidar_cone_array_files.sort()
    for lidar_cone_array_file in lidar_cone_array_files:
        lidar_cone_array_file = os.path.join(lidar_cone_array_folder,
                                             lidar_cone_array_file)
        lidar_cone_arrays.append(
            np.fromfile(lidar_cone_array_file).reshape(-1, 3))

    # Filter comparable data
    cone_array_types = ["gt_left", "gt_right", "lidar"]
    cone_arrays_dict = {}
    for cone_array_type in cone_array_types:
        cone_arrays_dict[cone_array_type] = []
        if cone_array_type == "gt_left":
            for idx in indices:
                cone_arrays_dict[cone_array_type].append(
                    gt_cone_arrays_dict["left"][idx])
        elif cone_array_type == "gt_right":
            for idx in indices:
                cone_arrays_dict[cone_array_type].append(
                    gt_cone_arrays_dict["right"][idx])
        elif cone_array_type == "lidar":
            for idx in indices:
                cone_arrays_dict[cone_array_type].append(
                    lidar_cone_arrays[idx])
        else:
            sys.exit()

    return cone_arrays_dict


def visualize_cone_arrays(cfg):
    cone_arrays_dict = match_cone_arrays(cfg)


def calculate_metrics(cfg):
    pass


def main():
    cfg = command_line_parser()

    if cfg.mode == "vis":
        visualize_cone_arrays(cfg)
    elif cfg.mode == "metrics":
        calculate_metrics(cfg)
    else:
        print("Moop")


if __name__ == "__main__":
    main()
