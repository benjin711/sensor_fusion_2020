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
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Preprocess image and point cloud data.')

    parser.add_argument('-r',
                        '--rosbag_data_folder_path',
                        type=str,
                        required=True,
                        help='Specify rosbag data folder path')

    parser.add_argument('--match_images', dest='match_images', action='store_true', help='Run algorithm to match images according to their timestamps to triplets')

    cfg = parser.parse_args()

    return cfg
