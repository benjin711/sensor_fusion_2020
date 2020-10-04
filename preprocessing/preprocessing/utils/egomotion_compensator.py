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
        self.T = read_static_transformations(data_folder_path)

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
                               reference_timestamp, motion_compensation):
        """
        Transforms points from fw_lidar or mrh_lidar frame to the egomotion frame
        at the time of the reference_timestamp
        """
        point_cloud = read_point_cloud(point_cloud_file)

        # Determine static T_fw_mrh
        T_fw_mrh = self.T["fw_lidar_to_mrh_lidar"]

        # Determine static T_mrh_ego
        T_mrh_ego = self.T["mrh_lidar_to_egomotion"]

        # Determine T_ego_wor(point_timestamps)
        T_ego_wor = np.zeros((point_cloud.shape[0], 4, 4))
        rots = self.R_slerp_ego_wor(point_cloud[:, 4]).as_matrix()
        trans = self.t_interpolator_ego_wor(point_cloud[:, 4])
        T_ego_wor[:, :3, :3] = rots
        T_ego_wor[:, :3, 3] = trans
        T_ego_wor[:, 3, 3] = 1

        # Determine T_wor_ego(reference_timestamp)
        T_ego_wor_ref_t = np.zeros((4, 4))
        T_ego_wor_ref_t[:3, :3] = self.R_slerp_ego_wor(
            reference_timestamp).as_matrix()
        T_ego_wor_ref_t[:3,
                        3] = self.t_interpolator_ego_wor(reference_timestamp)
        T_ego_wor_ref_t[3, 3] = 1
        T_wor_ego = np.linalg.inv(T_ego_wor_ref_t)

        # Determine static T_ego_mrh
        T_ego_mrh = np.linalg.inv(self.T["mrh_lidar_to_egomotion"])

        if src_frame == 'fw_lidar':

            if motion_compensation:
                # Transform each point from fw_lidar frame to the world frame
                # and then back to the mrh frame at reference time
                T_fw_world_mrh = T_ego_mrh @ T_wor_ego @ T_ego_wor @ T_mrh_ego @ T_fw_mrh

                points = np.hstack(
                    (point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))))
                points = (T_fw_world_mrh @ np.expand_dims(points, axis=2))
                point_cloud[:, :3] = np.squeeze(points)[:, :3]
            else:
                points = np.hstack(
                    (point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))))
                points = (T_fw_mrh @ points.T).T
                point_cloud[:, :3] = points[:, :3]

        elif src_frame == 'mrh_lidar':

            if motion_compensation:
                # Transform each point from mrh_lidar frame to the world frame
                # and then back to the mrh frame at reference time
                T_mrh_world_mrh = T_ego_mrh @ T_wor_ego @ T_ego_wor @ T_mrh_ego

                # Bring all points from mrh_lidar to the egomotion frame
                points = np.hstack(
                    (point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))))
                points = (T_mrh_world_mrh @ np.expand_dims(points, axis=2))
                point_cloud[:, :3] = np.squeeze(points)[:, :3]
            else:
                pass  # dont do shit

        else:
            print("Source frame not supported.")
            sys.exit()

        # Write the point cloud back to file
        write_point_cloud(point_cloud_file, point_cloud)
