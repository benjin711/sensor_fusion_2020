import os
from utils.utils import get_camera_timestamps, timestamps_within_interval
import numpy as np


class DataPreprocesser:
    def __init__(self, cfg):
        self.data_folder_path = cfg.data_folder_path

        # Constants
        self.MIN_DTIMESTAMP_THRESHOLD = 0.01

    def match_images(self):
        # Read in the timestamp files
        timestamp_arrays_dict = get_camera_timestamps(self.data_folder_path)

        # For every forward camera image timestamp, find the closest
        # timestamps from the left and right camera images
        idx_triples = []
        new_reference_timestamps = []

        for forward_camera_timestamp_idx, forward_camera_timestamp in enumerate(
                timestamp_arrays_dict["forward_camera"]):

            left_camera_dtimestamps = np.abs(
                timestamp_arrays_dict["left_camera"] -
                forward_camera_timestamp)
            left_camera_min_dtimestamp_idx = np.argmin(left_camera_dtimestamps)
            left_camera_min_dtimestamp = left_camera_dtimestamps[
                left_camera_min_dtimestamp_idx]

            right_camera_dtimestamps = np.abs(
                timestamp_arrays_dict["right_camera"] -
                forward_camera_timestamp)
            right_camera_min_dtimestamp_idx = np.argmin(
                right_camera_dtimestamps)
            right_camera_min_dtimestamp = right_camera_dtimestamps[
                right_camera_min_dtimestamp_idx]

            if left_camera_min_dtimestamp < self.MIN_DTIMESTAMP_THRESHOLD and right_camera_min_dtimestamp < self.MIN_DTIMESTAMP_THRESHOLD:

                left_camera_timestamp = timestamp_arrays_dict["left_camera"][
                    left_camera_min_dtimestamp_idx]
                right_camera_timestamp = timestamp_arrays_dict["right_camera"][
                    right_camera_min_dtimestamp_idx]

                if timestamps_within_interval(
                        self.MIN_DTIMESTAMP_THRESHOLD,
                    (forward_camera_timestamp, left_camera_timestamp,
                     right_camera_timestamp)):

                    # Add the indices of the valid triple to a list
                    current_idx_triple = [
                        forward_camera_timestamp_idx,
                        left_camera_min_dtimestamp_idx,
                        right_camera_min_dtimestamp_idx
                    ]
                    idx_triples.append(current_idx_triple)

                    new_reference_timestamps.append(
                        np.mean([
                            forward_camera_timestamp, left_camera_timestamp,
                            right_camera_timestamp
                        ]))

        print("done")
