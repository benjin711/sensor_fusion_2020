import argparse

def command_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-b',
        '--base_folder',
        default='/media/benjin/Samsung_T5/AMZ/sensor_fusion_data',
        type=str,
        help='Specify local path of the sensor_fusion_data folder')

    parser.add_argument(
        '-l',
        '--label_paths_file',
        default=
        '/home/benjin/Development/git/sensor_fusion_2020/data/full/test.txt',
        type=str,
        help='Specify which data split to calculate metrics on')

    parser.add_argument('-m',
                        '--mode',
                        default='metrics',
                        type=str,
                        choices=['vis, metrics'],
                        help='Specify what the program should do')

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