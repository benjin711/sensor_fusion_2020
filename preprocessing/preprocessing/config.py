import argparse
import json


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def command_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Preprocess image and point cloud data.')

    parser.add_argument(
        '--preprocess_all',
        dest='preprocess_all',
        action='store_true',
        help='Extract all rosbags in the base_folder and its subdirectories')

    parser.add_argument('-b',
                        '--base_folder',
                        default="",
                        type=str,
                        help='Specify base folder where all data is stored')

    parser.add_argument('-d',
                        '--data_folder_path',
                        type=str,
                        default="",
                        help='Specify rosbag data folder path')

    parser.add_argument(
        '-k',
        '--keep_orig_data_folders',
        dest='keep_orig_data_folders',
        type=str2bool,
        default=True,
        help=
        'Keep the original data folders (which are not grouped by timestamp)')

    parser.add_argument(
        '--match_data',
        dest='match_data',
        action='store_true',
        help=
        'Run algorithm to match images, point clouds, cone positions and car positions according to their timestamps'
    )

    parser.add_argument('-m',
                        '--motion_compensation',
                        type=str2bool,
                        default=True,
                        help='Motion compensate point clouds')

    parser.add_argument('--generate_rgbd',
                        dest='generate_rgbd',
                        action='store_true',
                        help='Generate RGBD images')

    parser.add_argument(
        '-i',
        '--icp_rots',
        type=str2bool,
        default=False,
        help='Calculate relative rotations between consecutive point clouds')

    cfg = parser.parse_args()

    return cfg
