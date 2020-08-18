import open3d as o3d
from utils import read_point_cloud

pc = read_point_cloud(
    "/media/benjin/Windows/Users/benja/Data/amz_sensor_fusion_data/2020-07-05_tuggen/data/autocross_2020-07-05-12-35-31/fw_lidar_filtered/00000000.bin"
)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc[:, :3])

o3d.visualization.draw_geometries([pcd])
