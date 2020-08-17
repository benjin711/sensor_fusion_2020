from config import command_line_parser
from data_preprocesser import DataPreprocesser
import pickle


def main(cfg):
    # data_preprocesser_instance = DataPreprocesser(cfg)

    # Match images to triplets and generate the corresponding reference timestamps
    # Then match point clouds to the triplets to create quintuples
    # if cfg.match_data:
    #     if cfg.perfect_data:
    #         data_preprocesser_instance.match_images_1()
    #     else:
    #         data_preprocesser_instance.match_images_2()

    # Dump data_preprocessor
    # with open('./preprocessing/preprocessing/data_preprocessor_instance.pkl',
    #           'wb') as output_pkl:
    #     pickle.dump(data_preprocesser_instance,
    #                 output_pkl,
    #                 protocol=pickle.HIGHEST_PROTOCOL)

    # THIS COMMENTED CODE DOESN'T WORK BUT THE LINES BELOW DO ?!?!
    # Load data_preprocessor
    # with open(
    #         '/home/benjin/Development/git/sensor_fusion_2020/preprocessing/preprocessing/data_preprocessor_instance.pkl',
    #         'rb') as input_pkl:
    #     data_preprocessor_instance = pickle.load(input_pkl)

    data_preprocesser_instance = pickle.load(
        open(
            '/home/benjin/Development/git/sensor_fusion_2020/preprocessing/preprocessing/data_preprocessor_instance.pkl',
            'rb'))

    data_preprocesser_instance.match_point_clouds()


if __name__ == "__main__":
    cfg = command_line_parser()
    main(cfg)
