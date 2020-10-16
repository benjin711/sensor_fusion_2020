import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from utils.lid_cam_calibration import *


def commandline_parser():
    parser = argparse.ArgumentParser(
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--data_root',
        type=str,
        default='',
        required=True,
        help='Path to top level directory containing data, gtmd, rosbag, etc')
    parser.add_argument('--calib_dir',
                        type=str,
                        default='',
                        required=True,
                        help='Path to directory containing camera infos')

    cfg = parser.parse_args()

    return cfg


def draw_depth(points, image, R, t, rect, check_depth=False):
    '''
    Draw points within corresponding camera's FoV on image provided.
    If no image provided, points are drawn on an empty(black) background.
    '''
    depth_layer = np.zeros(image.shape[:-1])

    pixels = to_pixel(points, R, t, rect=rect)
    for i in range(pixels.shape[1]):
        if points[i, 1] > 0:
            continue
        if ((pixels[0, i] < 0) | (pixels[1, i] < 0) |
            (pixels[0, i] > image.shape[1]) | (pixels[1, i] > image.shape[0])):
            continue
        depth_layer[np.int32(pixels[1, i]),
                    np.int32(pixels[0, i])] = np.sqrt(
                        np.sum(np.power(points[i, :], 2)))
    depth_layer = np.expand_dims(depth_layer, 2)
    mask_layer = (depth_layer > 0).astype(np.float)
    if check_depth:
        cv2.imshow('depth_mask', mask_layer)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()
    rgbd = np.concatenate((image.astype(np.float), depth_layer, mask_layer),
                          axis=-1)

    return rgbd


def main():
    cfg = commandline_parser()
    data_path = os.path.join(cfg.data_root, 'data')
    bag_names = glob(os.path.join(data_path, '*'))
    intrinsics_path = cfg.calib_dir
    extrinsics_path = os.path.join(cfg.data_root, 'extrinsics')
    if not os.path.exists(extrinsics_path):
        print('No extrinsic calibration for this test day')
        print('Please use preprocessing/preprocessing/' + \
              'utils/lid_cam_calibration.py to calibrate first')
        exit()
    output_path = os.path.join(cfg.data_root, 'rgbd')
    os.makedirs(output_path, exist_ok=True)

    cameras = ['left', 'forward', 'right']
    for bag in bag_names:
        bag_name = os.path.basename(bag)
        output_path_local = os.path.join(output_path, bag_name)
        pc_fw_files = sorted(
            glob(
                os.path.join(data_path, bag_name, 'fw_lidar_filtered',
                             '*.bin')))
        pc_mrh_files = sorted(
            glob(
                os.path.join(data_path, bag_name, 'mrh_lidar_filtered',
                             '*.bin')))

        for cam in cameras:
            output_path_local_cam = os.path.join(output_path_local, cam)
            os.makedirs(output_path_local_cam, exist_ok=True)
            K_mtx, distort_coef = create_pinhole_camera(
                os.path.join(intrinsics_path, cam + '.yaml'))

            img_files = sorted(
                glob(
                    os.path.join(data_path, bag_name, cam + '_camera_filtered',
                                 '*.png')))
            extrinsic_file = os.path.join(extrinsics_path,
                                          f'extrinsics_lidar_{cam}.yaml')
            if not os.path.exists(extrinsic_file):
                print(f'No extrinsic calibration for {cam} camera.')
                print('Please use preprocessing/preprocessing/' + \
                      'utils/lid_cam_calibration.py to calibrate first')
                exit()
            fs_read = cv2.FileStorage(extrinsic_file, cv2.FILE_STORAGE_READ)
            R = fs_read.getNode('R_mtx').mat()
            t = fs_read.getNode('t_mtx').mat()
            print(f'Generating RGBD images for {cam} camera:')
            for idx in tqdm(range(len(img_files))):
                pc = np.concatenate(
                    ((np.fromfile(pc_fw_files[idx],
                                  dtype=np.float64).reshape(-1, 6))[:, :3],
                     (np.fromfile(pc_mrh_files[idx],
                                  dtype=np.float64).reshape(-1, 6))[:, :3]))

                img = cv2.undistort(cv2.imread(img_files[idx]),
                                    cameraMatrix=K_mtx,
                                    distCoeffs=distort_coef)
                basename = os.path.splitext(os.path.basename(
                    img_files[idx]))[0]

                rgbd = draw_depth(pc, img, R, t, K_mtx, check_depth=False)
                rgbd.tofile(
                    os.path.join(output_path_local_cam, basename + '.bin'))


if __name__ == '__main__':
    main()
