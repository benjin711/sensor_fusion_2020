import os
import open3d as o3d
import numpy as np
import yaml
import cv2
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from mpl_toolkits.mplot3d import Axes3D
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def commandline_parser():
    parser = argparse.ArgumentParser(
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_root',
                        type=str,
                        default='',
                        required=True,
                        help='Path to top level directory containing data, gtmd, rosbag, etc')
    parser.add_argument('--calib_dir',
                        type=str,
                        default='',
                        required=True,
                        help='Path to directory containing camera infos')
    parser.add_argument('--lidar',
                        type=str,
                        default='fw',
                        required=True,
                        help='LiDAR to Calibrate: [fw, mrh]')
    parser.add_argument('--camera',
                        type=str,
                        default='',
                        required=True,
                        help='Camera to Calibrate: [left, right, forward]')
    parser.add_argument('--check_projection',
                        type=str2bool,
                        default=False,
                        required=False,
                        help='Check projection by visualization through video')
    cfg = parser.parse_args()

    return cfg


def create_pcd(xyz, scores=None, vmin=None, vmax=None, cmap=cm.tab20c):
    if vmin is None:
        vmin = np.min(scores)
    if vmax is None:
        vmax = np.max(scores)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = np.asarray(mapper.to_rgba(scores))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd


def create_pinhole_camera(file):
    with open(file) as f:
        try:
            data = yaml.load(f, Loader=yaml.CLoader)
        except AttributeError:
            data = yaml.load(f, Loader=yaml.Loader)

    intrinsic_matrix = np.array(data['camera_matrix']['data']).reshape(
        (data['camera_matrix']['rows'], data['camera_matrix']['cols']))
    distortion = np.array(data['distortion_coefficients']['data']).reshape(
        (data['distortion_coefficients']['rows'],
         data['distortion_coefficients']['cols']))

    return intrinsic_matrix, distortion


def select_img_points(img):
    ref_pt = np.zeros((2,))
    img_points = []

    def correspondence_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ref_pt[0], ref_pt[1] = x, y

    cv2.namedWindow("Correspondences")
    cv2.setMouseCallback("Correspondences", correspondence_cb)

    while True:
        curr_img = cv2.circle(img.copy(),
                              tuple(ref_pt.astype(np.uint)),
                              radius=2,
                              color=(255, 0, 0),
                              thickness=-1)
        cv2.imshow("Correspondences", curr_img)
        key = cv2.waitKey(1)
        if key == ord('y'):
            img_points.append(ref_pt.copy())
        elif key == ord('q'):
            if len(img_points) < 3:
                print("At least 3 points needed.")
            else:
                cv2.destroyAllWindows()
                break
    return np.array(img_points)


def pick_points(pcd:o3d.geometry.PointCloud):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift & right click] to undo point picking")
    print("   Press [shift & +/-] to change the size of the ball")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def to_pixel(pcd:np.array, R:np.array, T:np.array, rect:np.array):
    '''
    Generate pixel coordinate for all points
    '''
    one_mat = np.ones((pcd.shape[0], 1))
    point_cloud = np.concatenate((pcd, one_mat), axis=1)

    transformation = np.hstack((R, T))

    # Project point into Camera Frame
    point_cloud_cam = np.matmul(transformation, point_cloud.T)

    # Remove the Homogenious Term
    point_cloud_cam = np.matmul(rect, point_cloud_cam)

    # Normalize the Points into Camera Frame
    pixels = point_cloud_cam[::] / point_cloud_cam[::][-1]
    pixels = np.delete(pixels, 2, axis=0)
    return pixels


def depth_color(points, min_d=0, max_d=30):
    """
    print Color(HSV's H value) corresponding to distance(m)
    close distance = red , far distance = blue
    """
    dist = np.sqrt(
        np.add(np.power(points[:, 0], 2), np.power(points[:, 1], 2),
               np.power(points[:, 2], 2)))
    np.clip(dist, 0, max_d, out=dist)

    return (((dist - min_d) / (max_d - min_d)) * 128).astype(np.uint8)


def draw_points(points, image, R, t, rect=np.eye(3)):
    '''
    Draw points within corresponding camera's FoV on image provided.
    If no image provided, points are drawn on an empty(black) background.
    '''
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color = depth_color(points)
    pixels = to_pixel(points, R, t, rect=rect)
    for i in range(pixels.shape[1]):
        if points[i, 1] > 0:
            continue
        if ((pixels[0, i] < 0) | (pixels[1, i] < 0) |
            (pixels[0, i] > hsv_image.shape[1]) |
            (pixels[1, i] > hsv_image.shape[0])):
            continue
        cv2.circle(hsv_image, (np.int32(pixels[0, i]), np.int32(pixels[1, i])),
                   2, (int(color[i]), 255, 255), -1)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


if __name__ == '__main__':
    pwd = os.path.abspath(".")
    cfg = commandline_parser()
    pcfiles = sorted(
        glob(os.path.join(cfg.data_root,
                          'data', '*',
                          cfg.lidar + '_lidar_filtered',
                          '*.bin')))
    imgfile = sorted(
        glob(
            os.path.join(cfg.data_root,
                         'data', '*',
                         cfg.camera + '_camera_filtered',
                         '*.png')))

    for i in range(len(imgfile)):
        img = cv2.imread(imgfile[i])
        if np.sum(img) > 0:
            break
    pc = (np.fromfile(pcfiles[i], dtype=np.float64).reshape(-1, 6))[:, :4]

    K_mtx, distort = create_pinhole_camera(
        os.path.join(cfg.calib_dir, cfg.camera + '.yaml'))

    img_points = select_img_points(img)
    pcd = create_pcd(pc[:, :3], pc[:, 3])
    pc_index = pick_points(pcd)

    _, R, t = cv2.solvePnP(pc[pc_index, :3], img_points, K_mtx, distort)
    R = cv2.Rodrigues(R)[0]
    t = t.reshape((3, 1))
    print('Calibration Result:')
    print(f'rotation matrix: {R}')
    print(f'translation vector: {t}')

    img = cv2.undistort(img, cameraMatrix=K_mtx, distCoeffs=distort)
    projection = draw_points(pc[:, :3], img, R, t, K_mtx)
    cv2.imshow('Projection', projection)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    extrinsics_path = os.path.join(cfg.data_root, 'extrinsics')
    os.makedirs(extrinsics_path, exist_ok=True)
    f_name = os.path.join(extrinsics_path,
                          f'extrinsics_lidar_{cfg.camera}.yaml')
    fs_write = cv2.FileStorage(f_name, cv2.FILE_STORAGE_WRITE)
    fs_write.write('R_mtx', R)
    fs_write.write('t_mtx', t)

    if cfg.check_projection:
        fs_read = cv2.FileStorage(os.path.join(
                    pwd, 'extrinsics',
                    f'extrinsics_{cfg.lidar}_{cfg.camera}.npz'), cv2.FILE_STORAGE_READ)
        R = fs_read.getNode('R_mtx').mat() 
        t = fs_read.getNode('t_mtx').mat() 

        print(f'rotation vector: {R.squeeze()}')
        print(f'translation vector: {t.squeeze()}')

        out = cv2.VideoWriter(f'{cfg.lidar}_{cfg.camera}.mp4',
                              cv2.VideoWriter_fourcc(*'MP4V'), 10.0,
                              (img.shape[1], img.shape[0]))
        for i in range(len(imgfile)):
            img = cv2.imread(imgfile[i])
            pc = (np.fromfile(pcfiles[i], dtype=np.float64).reshape(-1,
                                                                    6))[:, :4]
            img = cv2.undistort(img, cameraMatrix=K_mtx, distCoeffs=distort)
            projection = draw_points(pc[:, :3], img, R, t, K_mtx)
            out.write(projection)
        out.release()
        print(f'Generated Projection Video {cfg.lidar}_{cfg.camera}.mp4')
