import sensor_msgs.point_cloud2 as pc2
import numpy as np


def convert_msg_to_numpy(pc_msg):
    pc = []
    for point in pc2.read_points(pc_msg, skip_nans=True):
        pc.append(point)
    return np.array(pc)
