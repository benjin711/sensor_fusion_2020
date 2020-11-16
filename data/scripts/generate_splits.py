"""Given the base directory, find all labels. According to the
percentages given for train, val, test, split the labels accordingly.
Generate train.txt, val.txt, and test.txt with the paths to each label."""

from glob import glob
import os
import random
import argparse

def generate_splits(cfg):
    base_dir, output_dir, train_ratio, val_ratio, test_ratio, total_num = \
    cfg.base_dir, cfg.output_dir, cfg.train_ratio, cfg.val_ratio, \
    cfg.test_ratio, cfg.total_num

    label_files = sorted(glob(os.path.join(base_dir, '*', 'data', '*',
                                                '*_labels', '*.txt')))
    random.shuffle(label_files)

    if 0 < total_num < len(label_files):
        label_files = label_files[:total_num]

    if train_ratio + val_ratio + test_ratio != 1.00:
        print("[ERROR] Train, val, and test ratio need to sum to one.")
        exit()

    train_idxs = list(range(0, round(len(label_files)*train_ratio)))
    val_idxs = list(range(train_idxs[-1], train_idxs[-1] + 1 + round(len(label_files)*val_ratio)))
    test_idxs = list(range(val_idxs[-1], len(label_files)))

    os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'train.txt'), 'w') as train_f:
        for idx in train_idxs:
            train_f.write(f"{label_files[idx]}\n")

    with open(os.path.join(output_dir, 'val.txt'), 'w') as val_f:
        for idx in val_idxs:
            val_f.write(f"{label_files[idx]}\n")

    with open(os.path.join(output_dir, 'test.txt'), 'w') as test_f:
        for idx in test_idxs:
            test_f.write(f"{label_files[idx]}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, help='Base directory with test days.')
    parser.add_argument('--output-dir', type=str, help='Directory to output train.txt, val.txt, and test.txt.')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Percentage of labels to use for training (<=1.00)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Percentage of labels to use for validation (<=1.00)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Percentage of labels to use for testing (<=1.00)')
    parser.add_argument('--total-num', type=int, default=-1, help='Total number of labels to use. Default is all.')
    args = parser.parse_args()
    generate_splits(args)



