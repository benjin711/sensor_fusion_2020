# The intent of this script is to create a data split file (like test.txt, val.txt) that 
# takes all paths from test.txt and only retains the paths of the left or right camera
# that have a respective right or left pair. So this script basically searches for left right
# pairs in the test.txt file for the model to do inference on
# Ok, so while this is a good idea, it was not done this way.. ups, see below

import os

path = "/cluster/home/benjin/git/sensor_fusion_2020/data/full/test.txt"
new_file_name = "test_sf.txt"

cameras = ["forward", "left", "right"]

with open(path, "r") as fr:
    orig_paths = fr.readlines()

with open(os.path.join(os.path.dirname(path), new_file_name), "w") as fw:
    for orig_path in orig_paths:
        which_camera = [camera for camera in cameras if orig_path.find(camera) > 0][0]

        if which_camera == "forward":
            continue

        r_path = orig_path.replace(which_camera, "right").rstrip("\n")
        l_path = orig_path.replace(which_camera, "left").rstrip("\n")

        if os.path.exists(r_path) and os.path.exists(l_path):
            
            fw.write(r_path + "\n")
            fw.write(l_path + "\n")


# The original version of this script that nobody should know about
# with open(path, "r") as fr:
#     orig_paths = fr.readlines()

# with open(os.path.join(os.path.dirname(path), new_file_name), "w") as fw:
#     for orig_path in orig_paths:
#         which_camera = [camera for camera in cameras if orig_path.find(camera) > 0][0]

#         if which_camera == "forward":
#             continue

#         for camera in ["right", "left"]:
#             if os.path.exists(orig_path.replace(which_camera, camera).rstrip("\n")):
#                 fw.write(orig_path.replace(which_camera, camera) + "\n")