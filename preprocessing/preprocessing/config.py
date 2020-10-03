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

    parser.add_argument('-d',
                        '--data_folder_path',
                        type=str,
                        required=True,
                        help='Specify rosbag data folder path')

    parser.add_argument(
        '-k',
        '--keep_orig_data_folders',
        dest='keep_orig_data_folders',
        action='store_true',
        help='Keep the original data folders (not grouped by timestamp)')

    parser.add_argument(
        '--match_data',
        dest='match_data',
        action='store_true',
        help=
        'Run algorithm to match images and point clouds according to their timestamps to quintests'
    )

    parser.add_argument(
        '-p',
        '--perfect_data',
        dest='perfect_data',
        action='store_true',
        help='Discard timestamps when at least one camera failed ')

    parser.add_argument('-m',
                        '--motion_compensation',
                        type=str2bool,
                        default=True,
                        help='Specify rosbag data folder path')

    cfg = parser.parse_args()

    return cfg
