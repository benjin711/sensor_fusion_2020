from config import command_line_parser
from data_preprocesser import DataPreprocesser
import pickle
import os
import sys


def main():
    cfg = command_line_parser()

    # Match images to triplets and generate the corresponding reference timestamps
    # Then match point clouds, car RTK data and cone data to the triplets to create septuples

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

    for idx, data_folder_path in enumerate(data_folder_paths):
        print("\nData folder {}: {}/{}".format(
            os.path.basename(data_folder_path), idx + 1,
            len(data_folder_paths)))

        cfg.data_folder_path = data_folder_path

        data_preprocesser_instance = DataPreprocesser(cfg)
        if cfg.match_data:
            if not cfg.perfect_data:
                data_preprocesser_instance.match_data_step_1()
            else:
                data_preprocesser_instance.match_images_perfect_data()

            data_preprocesser_instance.match_data_step_2(
                cfg.motion_compensation)

    # Dump data_preprocessor
    # with open('./data_preprocessor_instance.pkl',
    #           'wb') as output_pkl:
    #     pickle.dump(data_preprocesser_instance,
    #                 output_pkl,
    #                 protocol=pickle.HIGHEST_PROTOCOL)

    # THIS COMMENTED CODE DOESN'T WORK BUT THE LINES BELOW DO ?!?!
    # Load data_preprocessor
    # with open(
    #         '/home/benjin/Development/git/sensor_fusion_2020/preprocessing/preprocessing/data_preprocessor_instance.pkl',
    #         'rb') as input_pkl:
    #     data_preprocessor_instance = pickle.load(input_pkl)

    # data_preprocesser_instance = pickle.load(
    #     open(
    #         '/home/benjin/Development/git/sensor_fusion_2020/preprocessing/preprocessing/data_preprocessor_instance.pkl',
    #         'rb'))

    # data_preprocesser_instance.match_point_clouds()


if __name__ == "__main__":
    main()
