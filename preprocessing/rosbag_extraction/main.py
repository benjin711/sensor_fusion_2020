from config import command_line_parser
from rosbag_extractor import RosbagExtractor
import pickle
import os


def main(cfg):

    if cfg.debug:
        # Load object from pickles folder
        with open('pickles/extractor.pkl', 'rb') as input_pkl:
            extractor = pickle.load(input_pkl)
    else:
        extractor = RosbagExtractor(cfg.source)

        # Pickle this extractor so we don't need to create it all the time
        # Somehow not possible to pickle extractor..

        # if not os.path.isfile('pickles/extractor.pkl'):
        #     with open('pickles/extractor.pkl', 'wb') as output_pkl:
        #         pickle.dump(extractor, output_pkl, pickle.HIGHEST_PROTOCOL)

    for topic in cfg.topics:
        extractor.extract(topic)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    cfg = command_line_parser()
    main(cfg)
