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
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-r',
                        '--rosbag_file_path',
                        type=str,
                        required=True,
                        help='Specify rosbag file path')

    parser.add_argument('-m',
                        '--moving_only',
                        default=True,
                        type=str2bool,
                        help='Only extract the data when the car was moving')

    parser.add_argument(
        '-p',
        '--point_cloud_file_format',
        type=str,
        default='npy',
        choices=['npy', 'bin'],
        help='The point clouds can be extracted in numpys .npy or .bin format')

    parser.add_argument(
        '-t',
        '--topics',
        type=json.loads,
        default=
        '["/sensors/fw_lidar/point_cloud_raw", "/sensors/mrh_lidar/point_cloud_raw", "/sensors/right_camera/image_color", "/sensors/forward_camera/image_color", "/sensors/left_camera/image_color", "/tf", "/pilatus_can/GNSS"]',
        help='Specify the topics that should be extracted from')

    cfg = parser.parse_args()

    return cfg
