import glob
import os
import argparse
from tqdm import tqdm

import numpy as np

base_dir = '/media/carter/Samsung_T5/sensor_fusion_data_tmp_2'
def main(cfg):
    base_dir = cfg.base_dir
    paths = glob.glob(os.path.join(base_dir, '*', 'data', '*', '*_labels', '*.txt'))

    for p in tqdm(paths):
        data = np.genfromtxt(p) # cls, depth, xywh
        data_xy = data[:, [2, 3]]
        data_wh = data[:, [4, 5]]
        new_data_xy = data_xy + data_wh/2
        data[:, [2, 3]] = new_data_xy
        np.savetxt(p, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix incorrect XYWH labels "
                                                 "where the incorrect labels are have XY as "
                                                 "the top left, instead of the center of the box.")
    parser.add_argument('--base-dir', type=str, help="Top level directory to "
                                                     "glob for label data.")
    args = parser.parse_args()
    main(args)


