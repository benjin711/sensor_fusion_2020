from config import command_line_parser
from data_preprocesser import DataPreprocesser


def main(cfg):
    data_preprocesser = DataPreprocesser(cfg)

    # Match images to triplets and generate the corresponding reference timestamps
    if cfg.match_images:
        if cfg.perfect_data:
            data_preprocesser.match_images_1()
        else:
            data_preprocesser.match_images_2()


if __name__ == "__main__":
    cfg = command_line_parser()
    main(cfg)
