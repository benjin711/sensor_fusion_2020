import glob
import os
import argparse
from tqdm import tqdm

import numpy as np

def main(cfg):
    base_dir = cfg.base_dir
    start_idx = cfg.start_idx
    paths = glob.glob(os.path.join(base_dir, '*', 'data', '*', '*_labels', '*.txt'))

    for i, p in enumerate(tqdm(paths)):
        if i < start_idx:
            continue

        data = np.genfromtxt(p).reshape((-1, 6)) # cls, depth, xywh
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
    parser.add_argument('--start-idx', type=int, help="Index to start from")
    args = parser.parse_args()
    main(args)


