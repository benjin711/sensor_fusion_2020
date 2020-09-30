from utils.utils import *
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy import interpolate
import os
import sys


class EgomotionCompensator:
    """
    Provide point cloud that is not motion compensated
    Provide reference time stamp, source frame, 
    """

    def __init__(self, data_folder_path):
        self.T = read_static_transformations(
            data_folder_path)

        ego_wor_transformations = read_dynamic_transformation(
            "egomotion_to_world", data_folder_path)

        # Use slerp for interpolating between rotations and linear interpolation for translations
        key_rots = R.from_quat(ego_wor_transformations[:, 4:8])
        key_times = ego_wor_transformations[:, 0]
        self.R_slerp_ego_wor = Slerp(key_times, key_rots)
        self.t_interpolator_ego_wor = interpolate.interp1d(
            key_times,
            ego_wor_transformations[:, 1:4],
            axis=0,
            assume_sorted=True)

    def egomotion_compensation(self, point_cloud_file, src_frame,
                               reference_timestamp):
        """
        Transforms points from fw_lidar or mrh_lidar frame to the egomotion frame
        at the time of the reference_timestamp
        """
        point_cloud = read_point_cloud(point_cloud_file)

        T_ego_wor = np.zeros((4, 4))
        T_ego_wor[:3, :3] = self.R_slerp_ego_wor(
            reference_timestamp).as_matrix()
        T_ego_wor[:3, 3] = self.t_interpolator_ego_wor(reference_timestamp)
        T_ego_wor[3, 3] = 1
        T_wor_ego = np.linalg.inv(T_ego_wor)

        if src_frame == 'fw_lidar':
            # Transform each point from fw_lidar frame to the world frame
            # and then back to the mrh frame at reference time
            T_fw_ego = self.T["mrh_lidar_to_egomotion"] @ self.T["fw_lidar_to_mrh_lidar"]

            # Bring all points from fw_lidar to the egomotion frame
            points = np.hstack((point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))))
            point_cloud[:,:3] = (T_fw_ego @ points.T).T[:,:3]

            for idx in range(point_cloud.shape[0]):
                p_xyz = point_cloud[idx][:3]
                p_timestamp = point_cloud[idx][4]

                # TODO: Error in interpolation from discrepancy between
                #       ros msg timestamp, and the timestamp of individ. points

                T_ego_wor[:3, :3] = self.R_slerp_ego_wor(
                    p_timestamp).as_matrix()
                T_ego_wor[:3, 3] = self.t_interpolator_ego_wor(p_timestamp)
                T_ego_wor[3, 3] = 1

                p_xyz_new = T_ego_wor @ np.append(
                    p_xyz, 1)
                point_cloud[idx][:3] = p_xyz_new[:3]

            # Bring all points from world to mrh_lidar frame
            points = np.hstack((point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))))
            T_ego_mrh = np.linalg.inv(self.T["mrh_lidar_to_egomotion"])
            point_cloud[:,:3] = (T_ego_mrh @ T_wor_ego @ points.T).T[:,:3]
            

        elif src_frame == 'mrh_lidar':
            # Transform each point from mrh_lidar frame to the world frame
            # and then back to the mrh frame at reference time
            T_mrh_ego = self.T["mrh_lidar_to_egomotion"]

            # Bring all points from mrh_lidar to the egomotion frame
            points = np.hstack((point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))))
            point_cloud[:,:3] = (T_mrh_ego @ points.T).T[:,:3]

            for idx in range(point_cloud.shape[0]):
                p_xyz = point_cloud[idx][:3]
                p_timestamp = point_cloud[idx][4]

                T_ego_wor[:3, :3] = self.R_slerp_ego_wor(
                    p_timestamp).as_matrix()
                T_ego_wor[:3, 3] = self.t_interpolator_ego_wor(p_timestamp)
                T_ego_wor[3, 3] = 1

                p_xyz_new = T_ego_wor @ np.append(
                    p_xyz, 1)
                point_cloud[idx][:3] = p_xyz_new[:3]

            # Bring all points from world to mrh_lidar frame
            points = np.hstack((point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))))
            T_ego_mrh = np.linalg.inv(self.T["mrh_lidar_to_egomotion"])
            point_cloud[:,:3] = (T_ego_mrh @ T_wor_ego @ points.T).T[:,:3]

        else:
            print("Source frame not supported.")
            sys.exit()

        # Write the point cloud back to file
        write_point_cloud(point_cloud_file, point_cloud)
