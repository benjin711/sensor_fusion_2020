import glob
import os
import argparse
from tqdm import tqdm

import numpy as np
import cv2

def main(cfg):
    base_dir = cfg.base_dir
    im_paths = glob.glob(os.path.join(base_dir, '*', 'data', '*', '*_camera_filtered', '*.png'))
    di_paths = [x.replace('camera_filtered', 'di').replace('png', 'bin') for x in im_paths]
    m_paths = [x.replace('camera_filtered', 'm').replace('png', 'bin') for x in im_paths]

    max_depth = 0
    for i, im_p in enumerate(tqdm(im_paths)):
        m_p = m_paths[i]
        di_p = di_paths[i]
        im = cv2.imread(im_p)

        H, W, C = im.shape
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
    args = parser.parse_args()
    main(args)


