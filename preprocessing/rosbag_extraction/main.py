from config import command_line_parser
from rosbag_extractor import RosbagExtractor


def main(cfg):
    extractor = RosbagExtractor(cfg.rosbag_file_path)

    for topic in cfg.topics:
        extractor.extract(topic)

    if cfg.pickle_data:
        extractor.pickle_data()


if __name__ == "__main__":
    cfg = command_line_parser()
    main(cfg)
