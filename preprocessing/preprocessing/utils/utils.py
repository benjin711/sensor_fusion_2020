import numpy as np
import yaml
import os
import cv2
import shutil
from tqdm import tqdm
import sys
from scipy.spatial.transform import Rotation as R


def get_camera_timestamps(data_folder_path):
    timestamp_filepaths_dict = {
        "forward_camera":
        os.path.join(data_folder_path, "forward_camera/timestamps.txt"),
        "left_camera":
        os.path.join(data_folder_path, "left_camera/timestamps.txt"),
        "right_camera":
        os.path.join(data_folder_path, "right_camera/timestamps.txt")
    }

    return read_timestamps(timestamp_filepaths_dict)


def get_lidar_timestamps(data_folder_path):
    timestamp_filepaths_dict = {}

    if os.path.exists(os.path.join(data_folder_path,
                                   "fw_lidar/timestamps.txt")):
        timestamp_filepaths_dict["fw_lidar"] = os.path.join(
            data_folder_path, "fw_lidar/timestamps.txt")

    if os.path.exists(
            os.path.join(data_folder_path, "mrh_lidar/timestamps.txt")):
        timestamp_filepaths_dict["mrh_lidar"] = os.path.join(
            data_folder_path, "mrh_lidar/timestamps.txt")

    return read_timestamps(timestamp_filepaths_dict)


def get_image_size(data_folder_path):
    img = cv2.imread(
        os.path.join(data_folder_path, "forward_camera/00000000.png"))
    return img.shape


def read_timestamps(timestamp_filepaths_dict):
    timestamp_arrays_dict = {}
    for key in timestamp_filepaths_dict:
        timestamp_arrays_dict[key] = []

    for key in timestamp_filepaths_dict:
        with open(timestamp_filepaths_dict[key]) as timestamps_file:
            for timestamp in timestamps_file:
                timestamp_arrays_dict[key].append(timestamp)
            timestamp_arrays_dict[key] = np.array(timestamp_arrays_dict[key],
                                                  dtype=np.float)
    return timestamp_arrays_dict


def timestamps_within_interval(interval, timestamps):
    min_timestamp = np.min(np.array(timestamps))
    max_timestamp = np.max(np.array(timestamps))
    return max_timestamp - min_timestamp < interval


def write_reference_timestamps(dst_image_folder, reference_timestamps):
    print("\nUpdate timestamps.txt")
    with open(os.path.join(dst_image_folder, 'timestamps.txt'),
              'w') as filehandle:
        filehandle.writelines("{:.6f}\n".format(timestamp)
                              for timestamp in reference_timestamps)


def lists_in_dict_empty(dict_of_lists):
    total_num_elements = 0
    for key in dict_of_lists:
        total_num_elements += len(dict_of_lists[key])

    if total_num_elements != 0:
        return False
    else:
        return True


def read_point_cloud(point_cloud_file):
    _, ext = os.path.splitext(point_cloud_file)

    if ext == '.bin':
        point_cloud = (np.fromfile(point_cloud_file,
                                   dtype=np.float64).reshape(-1, 6))
    elif ext == '.npy':
        point_cloud = (np.load(point_cloud_file).astype(np.float64).reshape(
            -1, 6))
    else:
        print("Invalid point cloud format encountered.")
        sys.exit()

    return point_cloud


def write_point_cloud(point_cloud_file, point_cloud):
    _, ext = os.path.splitext(point_cloud_file)
    if ext == '.npy':
        np.save(point_cloud_file, point_cloud)
    elif ext == '.bin':
        point_cloud.tofile(point_cloud_file)
    else:
        print("Saving in specified point cloud format is not possible.")


def read_static_transformations(data_folder_path):

    static_transformation_file = os.path.join(
        data_folder_path,
        "../../static_transformations/static_transformations.yaml")

    with open(static_transformation_file, "r") as yaml_file:
        static_transformations = yaml.load(yaml_file, Loader=yaml.FullLoader)

    for _, item in static_transformations.items():
        static_transformations = item

    for name, transformation in static_transformations.items():
        yaw = transformation['rotation']['yaw']
        pitch = transformation['rotation']['pitch']
        roll = transformation['rotation']['roll']
        x = transformation['translation']['x']
        y = transformation['translation']['y']
        z = transformation['translation']['z']
        r = R.from_euler('zyx', [yaw, pitch, roll], degrees=False)
        t = np.array([x, y, z])

        T = np.zeros((4, 4))
        T[:3, :3] = r.as_matrix()
        T[:3, 3] = t
        T[3, 3] = 1

        static_transformations[name] = T

    return static_transformations


def read_dynamic_transformation(transform, data_folder_path):
    if transform == "egomotion_to_world":
        file_path = os.path.join(data_folder_path, "tf/egomotion_to_world.txt")

        return np.loadtxt(file_path, delimiter=",", skiprows=1)

    else:
        print("The requested dynamic transform doesn't exist")


def load_rosparam_mat(yaml_dict, param_name):
    """Given a dict intended for ROS, convert the list to a matrix for a given key"""
    rows = yaml_dict[param_name]['rows']
    cols = yaml_dict[param_name]['cols']
    data = yaml_dict[param_name]['data']

    return np.asarray(data).reshape((rows, cols))


def load_stereo_calib(camera_fn, stereo_fn):

    if not os.path.exists(camera_fn) or not os.path.exists(stereo_fn):
        print("Calibration files do not exist!")
        exit()

    calib = {}
    with open(camera_fn) as camera_f:
        camera_data = yaml.load(camera_f, Loader=yaml.FullLoader)

    with open(stereo_fn) as stereo_f:
        stereo_data = yaml.load(stereo_f, Loader=yaml.FullLoader)

    calib['camera_matrix'] = load_rosparam_mat(camera_data, 'camera_matrix')
    calib['distortion_coefficients'] = load_rosparam_mat(
        camera_data, 'distortion_coefficients')
    calib['image_width'] = camera_data['image_width']
    calib['image_height'] = camera_data['image_height']
    calib['rotation_matrix'] = load_rosparam_mat(stereo_data,
                                                 'rotation_matrix')
    calib['translation_vector'] = load_rosparam_mat(stereo_data,
                                                    'translation_vector')
    calib['fundamental_matrix'] = load_rosparam_mat(stereo_data,
                                                    'fundamental_matrix')

    return calib


def load_camera_calib(camera_fn):

    if not os.path.exists(camera_fn):
        print("Calibration file does not exist!")
        exit()

    calib = {}
    with open(camera_fn) as camera_f:
        try:
            camera_data = yaml.load(camera_f, Loader=yaml.CLoader)
        except AttributeError:
            camera_data = yaml.load(camera_f, Loader=yaml.Loader)

    calib['camera_matrix'] = load_rosparam_mat(camera_data, 'camera_matrix')
    calib['distortion_coefficients'] = load_rosparam_mat(
        camera_data, 'distortion_coefficients')
    calib['image_width'] = camera_data['image_width']
    calib['image_height'] = camera_data['image_height']
    calib['projection_matrix'] = load_rosparam_mat(camera_data,
                                                   'projection_matrix')

    return calib


def undistort_image(data_folder_path, image_path, camera):
    # Get image
    img = cv2.imread(image_path)

    # Get camera matrix and distortion coefficients
    calib_folder = os.path.join(data_folder_path, "..", "..",
                                "static_transformations")
    calib = load_camera_calib(
        os.path.join(calib_folder,
                     camera.split('_')[0] + '.yaml'))

    # Undistort image
    img = cv2.undistort(img,
                        cameraMatrix=calib['camera_matrix'],
                        distCoeffs=calib['distortion_coefficients'])

    # Write back to file
    cv2.imwrite(image_path, img)


def get_mrh_cam_transformations(static_transformations_folder, cameras):
    transformations_dict = {}

    calibration_files = os.listdir(static_transformations_folder)

    if all([
            "extrinsics_mrh_{}.yaml".format(camera) in calibration_files
            for camera in cameras
    ]):
        for camera in cameras:
            extrinsics_file = os.path.join(
                static_transformations_folder,
                "extrinsics_mrh_{}.yaml".format(camera))
            reader = cv2.FileStorage(extrinsics_file, cv2.FILE_STORAGE_READ)
            R = reader.getNode('R_mtx').mat()
            t = reader.getNode('t_mtx').mat()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.squeeze()
            transformations_dict[camera] = T
    else:
        print(
            "Extrinsic calibration between mrh -> cameras are missing in {static_calibration_folder}!"
        )
        sys.exit()

    return transformations_dict


def get_intrinsics(static_transformations_folder, cameras):
    """
    Returns a dict with the camera matrices corresponding to the
    cameras specified in the cameras list
    """
    K_dict = {}

    calibration_files = os.listdir(static_transformations_folder)

    if all(
        ["{}.yaml".format(camera) in calibration_files for camera in cameras]):
        for camera in cameras:
            camera_calib = load_camera_calib(
                os.path.join(static_transformations_folder,
                             "{}.yaml".format(camera)))
            K_dict[camera] = camera_calib['camera_matrix']
    else:
        print("Intrinsics are not available in {}!".format(
            static_transformations_folder))
        sys.exit()

    return K_dict


def get_pc_files(data_folder_path, lidars):
    """
    Returns a dict with the file paths to the pc files
    of the lidars specified by lidars and data_folder_path
    """
    pc_files_dict = {}

    for lidar in lidars:
        pc_folder = os.path.join(data_folder_path,
                                 "{}_lidar_filtered".format(lidar))

        if os.path.exists(pc_folder):
            pc_files = os.listdir(pc_folder)
            pc_files.remove("timestamps.txt")
            pc_files_dict[lidar] = sorted(
                [os.path.join(pc_folder, pc_file) for pc_file in pc_files])

    return pc_files_dict


def get_img_files(data_folder_path, cameras):
    """
    Returns a dict with the file paths to the img files
    of the cameras specified by camera and data_folder_path
    """
    img_files_dict = {}

    for camera in cameras:
        img_folder = os.path.join(data_folder_path,
                                  "{}_camera_filtered".format(camera))
        img_files = os.listdir(img_folder)
        img_files.remove("timestamps.txt")
        img_files_dict[camera] = sorted(
            [os.path.join(img_folder, img_file) for img_file in img_files])

    return img_files_dict


def project_points_to_pixels(pcd, R, t, K, img_shape):
    '''
    Generate pixel coordinates for all points and also
    calculate a mask which masks all valid pixels that come
    from points in front of the camera and lie within
    the image boundaries
    '''
    one_mat = np.ones((pcd.shape[0], 1))
    point_cloud = np.concatenate((pcd, one_mat), axis=1)

    transformation = np.hstack((R, t.reshape(3, 1)))

    # Transform points into the camera frame
    point_cloud_cam = np.matmul(transformation, point_cloud.T)

    # Ignore points behind the camera (z < 0)
    z_filter = point_cloud_cam[2, :] > 0
    pixels_cam = np.matmul(K, point_cloud_cam)

    # Normalize to pixels
    pixels_cam = pixels_cam[::] / pixels_cam[::][-1]
    pixels = np.delete(pixels_cam, 2, axis=0)

    # Ignore the ones outside of the image
    pixels = pixels.T
    greater_zero_filter = np.logical_and(pixels[:, 0] > 0, pixels[:, 1] > 0)
    smaller_image_shape_filter = np.logical_and(pixels[:, 0] < img_shape[1],
                                                pixels[:, 1] < img_shape[0])

    pixel_filter = np.logical_and(
        np.logical_and(greater_zero_filter, smaller_image_shape_filter),
        z_filter)

    return pixels, pixel_filter
