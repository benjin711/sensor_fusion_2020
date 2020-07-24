from config import command_line_parser
from rosbag_extractor import RosbagExtractor


def main(cfg):
    extractor = RosbagExtractor(cfg)

    for topic in cfg.topics:
        extractor.extract(topic)


if __name__ == "__main__":
    cfg = command_line_parser()
    main(cfg)
