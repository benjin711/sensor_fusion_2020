import argparse
import json


def command_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-s',
                        '--source',
                        type=str,
                        required=True,
                        help='Specify rosbag')

    parser.add_argument('-d',
                        '--debug',
                        type=bool,
                        default=False,
                        help='Debug flag')

    parser.add_argument('-m',
                        '--moving_only',
                        type=str,
                        default=True,
                        help='Only extract the data when the car was moving')

    parser.add_argument(
        '-t',
        '--topics',
        type=json.loads,
        default=
        '["/sensors/right_camera/image_color", "/sensors/forward_camera/image_color", "/sensors/left_camera/image_color", "/tf", "/pilatus_can/GNSS"]',  #"/sensors/fw_lidar/point_cloud_raw", "/sensors/mrh_lidar/point_cloud_raw", 
        help='Specify the topics that should be extracted from')

    cfg = parser.parse_args()

    return cfg
