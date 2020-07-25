import os
from utils.utils import get_camera_timestamps


class DataPreprocesser:
    def __init__(self, cfg):
        self.data_folder_path = cfg.data_folder_path

    def match_images(self):
        # Read in the timestamp files
        timestamp_arrays_dict = get_camera_timestamps()

        # For every forward camera image timestamp, find the closest
        # timestamps from the left and right camera images
        for idx, forward_camera_timestamp in enumerate(
                timestamp_arrays_dict["forward_camera"]):

            left_camera_timestamp_idx = np.argmin(
                np.abs(timestamp_arrays_dict["left_camera"] -
                       forward_camera_timestamp))

            right_camera_timestamp_idx = np.argmin(
                np.abs(timestamp_arrays_dict["right_camera"] -
                       forward_camera_timestamp))
