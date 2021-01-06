import os
from glob import glob
import subprocess

# 1) Get all the rosbag names that we need to download. Use the folders in the data folder to get these names.
# 2) Download from nas to the correct directories
BASE_FOLDER_LOCAL = "/media/benjin/Samsung_T5/AMZ/sensor_fusion_data"
ROSBAG_FOLDER_REMOTE = "amz@amz-nas.ethz.ch:/home/amz-nas/pilatus-2020/rosbags-small"

data_parent_folders = glob(os.path.join(BASE_FOLDER_LOCAL, "*", "data"))

for data_parent_folder in data_parent_folders:
    data_folders = next(os.walk(data_parent_folder))[1]
    rosbag_files = [data_folder + ".bag" for data_folder in data_folders]

    test_day_folder_local = data_parent_folder[:-5]
    rosbag_folder_local = os.path.join(test_day_folder_local, "rosbags")

    test_day_folder_remote = test_day_folder_local.replace(
        BASE_FOLDER_LOCAL, ROSBAG_FOLDER_REMOTE)

    rosbag_files = [
        os.path.join(test_day_folder_remote, rosbag_file)
        for rosbag_file in rosbag_files
    ]

    for rosbag_file in rosbag_files:
        subprocess.run([
            "rsync", "-hPr", "--ignore-existing", rosbag_file,
            rosbag_folder_local
        ])
