import open3d as o3d
import numpy as np
import os


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


fw_pc = read_point_cloud(
    "/media/benjin/Windows/Users/benja/Data/amz_sensor_fusion_data/2020-07-05_tuggen/data/autocross_2020-07-05-12-35-31/fw_lidar_filtered/00000010.bin"
)

mrh_pc = read_point_cloud(
    "/media/benjin/Windows/Users/benja/Data/amz_sensor_fusion_data/2020-07-05_tuggen/data/autocross_2020-07-05-12-35-31/mrh_lidar_filtered/00000010.bin"
)

fw_pcd = o3d.geometry.PointCloud()
fw_pcd.points = o3d.utility.Vector3dVector(fw_pc[:, :3])
fw_pcd.paint_uniform_color(np.array([255, 0, 0]))

mrh_pcd = o3d.geometry.PointCloud()
mrh_pcd.points = o3d.utility.Vector3dVector(mrh_pc[:, :3])
mrh_pcd.paint_uniform_color(np.array([0, 255, 0]))

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0, 0, 0])

o3d.visualization.draw_geometries([fw_pcd, mrh_pcd, mesh_frame])
