from config import command_line_parser
from data_preprocesser import DataPreprocesser
import pickle
import os
import sys


def get_data_folder_paths(cfg):
    data_folder_paths = []

    if cfg.preprocess_all:
        if not os.path.exists(cfg.base_folder):
            print("The specified base folder does not exist!")
            sys.exit()

        test_day_folders = os.listdir(cfg.base_folder)

        for d in test_day_folders:
            d = os.path.join(cfg.base_folder, d, "data")
            for data_folder in os.listdir(d):
                data_folder_paths.append(os.path.join(d, data_folder))

    else:
        if not os.path.exists(cfg.data_folder_path):
            print("The specified data folder path does not exist!")
            sys.exit()

        data_folder_paths.append(cfg.data_folder_path)

    return data_folder_paths


def main():
    cfg = command_line_parser()

    # For every time stamp have 3x images, 2x point clouds,
    # 1x cone positions, 1x car position
    data_folder_paths = get_data_folder_paths(cfg)

    for idx, data_folder_path in enumerate(data_folder_paths):
        print("\nData folder {}: {}/{}".format(
            os.path.basename(data_folder_path), idx + 1,
            len(data_folder_paths)))

        cfg.data_folder_path = data_folder_path

        data_preprocesser_instance = DataPreprocesser(cfg)
        if cfg.match_data:
            data_preprocesser_instance.match_data_step_1()
            data_preprocesser_instance.match_data_step_2(
                cfg.motion_compensation)

        if cfg.icp_rots:
            data_preprocesser_instance.extract_rotations()

        if cfg.generate_dim:
            data_preprocesser_instance.generate_dim()


if __name__ == "__main__":
    main()
