import glob
import os
import argparse
from tqdm import tqdm
import random

import numpy as np
import cv2

def main(cfg):
    base_dir, max_samples = cfg.base_dir, cfg.max_samples
    label_paths = glob.glob(os.path.join(base_dir, '*', 'data', '*', '*_labels', '*.txt'))
    random.shuffle(label_paths)

    im_paths = [x.replace('labels', 'camera_filtered').replace('png', 'bin') for x in label_paths]
    di_paths = [x.replace('labels', 'di').replace('png', 'bin') for x in label_paths]
    m_paths = [x.replace('labels', 'm').replace('png', 'bin') for x in label_paths]

    # Open one image to get dimensions. Assume same dimensions for all
    test_img = cv2.imread(im_paths[0])
    H, W, C = test_img.shape

    if max_samples < 0:
        max_samples = len(label_paths)

    max_depth = 0
    for i, im_p in enumerate(tqdm(im_paths)):
        if i > max_samples:
            break
        m_p = m_paths[i]
        di_p = di_paths[i]
        # im = cv2.imread(im_p)

        di = np.fromfile(di_p, dtype=np.float16).reshape((H, W, 2))
        d = di[:, :, 0]
        m = np.fromfile(m_p, dtype=np.bool).reshape((H, W, 1))

        curr_max = np.max(d)
        if curr_max > max_depth:
            max_depth = curr_max

    print(f"Max depth in dataset is {max_depth} m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterate through input data and determine statistics.")
    parser.add_argument('--base-dir', type=str, help="Top level directory to "
                                                     "glob for label data.")
    parser.add_argument('--max-samples', type=int, default=-1, help="Maximum number of input samples to analyze")
    args = parser.parse_args()
    main(args)


