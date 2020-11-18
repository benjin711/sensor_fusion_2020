from config import command_line_parser
from rosbag_extractor import RosbagExtractor
import os
import sys


def main():
    cfg = command_line_parser()

    rosbag_file_paths = []

    if cfg.extract_all:
        if not os.path.exists(cfg.base_folder):
            print("The specified base folder does not exist!")
            sys.exit()

        for root, dirs, files in os.walk(cfg.base_folder):
            for f in files:
                if(f.endswith(".bag")):
                    rosbag_file_paths.append(os.path.join(root, f))
    else:
        if not os.path.exists(cfg.rosbag_file_path):
            print("The specified rosbag_file_path does not exist!")
            sys.exit()

        rosbag_file_paths.append(cfg.rosbag_file_path)

    for idx, rosbag_file_path in enumerate(rosbag_file_paths):
        print("\nRosbag {}: {}/{}".format(os.path.basename(rosbag_file_path), idx+1,
                                          len(rosbag_file_paths)))

        cfg.rosbag_file_path = rosbag_file_path

        extractor = RosbagExtractor(cfg)

        ret = extractor.init_file_structure(cfg.topics)

        if ret:
            for topic in cfg.topics:
                extractor.extract(topic)


if __name__ == "__main__":
    main()
