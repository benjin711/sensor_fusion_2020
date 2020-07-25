from config import command_line_parser
from data_preprocesser import DataPreprocesser

def main(cfg):
  data_preprocesser = DataPreprocesser(cfg)

  # Match images two triplets and generate the corresponding reference timestamps
  data_preprocesser.match_images()


if __name__ == "__main__":
  cfg = command_line_parser()
  main(cfg)