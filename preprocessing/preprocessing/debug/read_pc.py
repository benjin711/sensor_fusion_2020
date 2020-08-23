import os
import numpy as np
import json


def read_point_cloud(point_cloud_file):
    _, ext = os.path.splitext(point_cloud_file)

    if ext == '.bin':
        point_cloud = (np.fromfile(point_cloud_file,
                                   dtype=np.float32).reshape(-1, 6))
    elif ext == '.npy':
        point_cloud = (np.load(point_cloud_file,
                               dtype=np.float32).reshape(-1, 6))
    else:
        print("Invalid point cloud format encountered.")
        sys.exit()

    return point_cloud


pc = read_point_cloud(
    "/media/benjin/Windows/Users/benja/Data/amz_sensor_fusion_data/2020-07-05_tuggen/data/autocross_2020-07-05-12-35-31/fw_lidar/00000600.bin"
)

print("done")

# 1593952640.989935
# 1593952710.689935
