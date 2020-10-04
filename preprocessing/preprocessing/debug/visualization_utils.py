import open3d as o3d
import numpy as np
import os
import sys


def read_point_cloud(point_cloud_file):
    _, ext = os.path.splitext(point_cloud_file)

    if ext == '.bin':
        point_cloud = (np.fromfile(point_cloud_file,
                                   dtype=np.float64).reshape(-1, 6))
    elif ext == '.npy':
        point_cloud = (np.load(point_cloud_file,
                               dtype=np.float64).reshape(-1, 6))
    else:
        print("Invalid point cloud format encountered.")
        sys.exit()

    return point_cloud


counter = 100

while True:

    fw_pc_filtered = read_point_cloud(
        "/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-07-12_tuggen/data/autocross_2020-07-12-09-54-31/mrh_lidar_filtered/"
        + str(counter).zfill(8) + ".bin")

    fw_pc_filtered_no_mc = read_point_cloud(
        "/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-07-12_tuggen/data/autocross_2020-07-12-09-54-31/mrh_lidar_filtered_no_mc/"
        + str(counter).zfill(8) + ".bin")

    fw_pcd_filtered = o3d.geometry.PointCloud()
    fw_pcd_filtered.points = o3d.utility.Vector3dVector(fw_pc_filtered[:, :3])
    fw_pcd_filtered.paint_uniform_color(np.array([1, 0, 0]))

    fw_pcd_filtered_no_mc = o3d.geometry.PointCloud()
    fw_pcd_filtered_no_mc.points = o3d.utility.Vector3dVector(
        fw_pc_filtered_no_mc[:, :3])
    fw_pcd_filtered_no_mc.paint_uniform_color(np.array([0, 1, 0]))

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0])

    o3d.visualization.draw_geometries(
        [fw_pcd_filtered_no_mc, fw_pcd_filtered, mesh_frame])

    counter += 1
