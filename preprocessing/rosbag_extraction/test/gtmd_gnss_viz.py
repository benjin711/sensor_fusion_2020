"""Test the parsing of GNSS and GTMD data by visualizing cone positions,
   vehicle positions, and vehicle heading."""

import numpy as np
from matplotlib import pyplot as plt
from cv2 import waitKey

from gtmd_parser import parse_gtmd_csv

gtmd_csv_path = "/media/carter/Samsung_T5/sensor-fusion-data/2020-07-05_tuggen/2020-07-05_tuggen.csv"
vehicle_poses_npy_path = '../../../vehicle_pos.npy'
skip_frames = 10
time_per_frame = 1

# Get cone positions, get extracted vehicle positions and headings
cone_xy = parse_gtmd_csv(gtmd_csv_path)
cone_xy = np.transpose(cone_xy)
vehicle_xy_heading = np.load(vehicle_poses_npy_path)

# Visualize each frame. Show each frame for 1 second.
for vehicle_pose_idx in range(0, vehicle_xy_heading.shape[0], skip_frames):
    plt.scatter(cone_xy[:, 0], cone_xy[:, 1], c='r')
    plt.scatter(vehicle_xy_heading[vehicle_pose_idx, 0],
                vehicle_xy_heading[vehicle_pose_idx, 1], c='g')
    heading = vehicle_xy_heading[vehicle_pose_idx, 2]
    dx = 2*np.cos(np.deg2rad(heading))
    dy = 2*np.sin(np.deg2rad(heading))
    plt.arrow(vehicle_xy_heading[vehicle_pose_idx, 0],
              vehicle_xy_heading[vehicle_pose_idx, 1],
              dx, dy)
    plt.show()
    waitKey(time_per_frame*1000)
