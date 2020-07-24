import sensor_msgs.point_cloud2 as pc2
import numpy as np


def convert_msg_to_numpy(pc_msg):
    pc = []
    for point in pc2.read_points(pc_msg, skip_nans=True):
        pc.append(point)
    return np.array(pc)


def get_driving_interval(egomotion_to_world_transform):
    timestamps_and_transforms = np.array(
        [element[:4] for element in egomotion_to_world_transform],
        dtype=np.float)
    timestamps = timestamps_and_transforms[:, 0]
    transforms = timestamps_and_transforms[:, 1:]

    # Get the time stamp when car first drove faster than 0.1m/s
    # Get the time stamp when car last drove faster than 0.1m/s
    # Equivalent to covering a distance of 0.1m/s * 0.005s (200Hz update rate)
    delta_distance = np.linalg.norm(transforms -
                                    np.roll(transforms, 1, axis=0),
                                    axis=1)
    # Skip first element
    tmp = delta_distance[1:] > 0.1 * 0.005

    start_idx = np.argmax(tmp)
    start_timestamp = timestamps[start_idx]

    end_idx = np.argmax(np.flip(tmp, 0))
    end_timestamp = timestamps[-1 - end_idx]

    return start_timestamp, end_timestamp
