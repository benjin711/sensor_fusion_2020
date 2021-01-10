import cv2
import numpy as np
import os
import copy
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt


def draw_depth(image, depth, window_name="Blank"):

    jet = cm = plt.get_cmap('hsv')
    # depth = np.sqrt(depth)
    vmax = np.max(depth)
    cNorm = colors.Normalize(vmin=0, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    new_image = image.copy()
    for r in range(depth.shape[0]):
        for c in range(depth.shape[1]):
            if depth[r, c] > 0:

                # new_image[r,
                #           c] = (np.array(scalarMap.to_rgba(depth[r, c])[:3]) *
                #                 255).astype(np.uint8)
                cv2.circle(new_image, (c, r),
                           radius=2,
                           color=tuple([
                               int(s * 255) for s in scalarMap.to_rgba(
                                   np.clip(depth[r, c], 0, vmax))[:3]
                           ]),
                           thickness=-1)

    cv2.imshow(window_name, new_image)
    cv2.waitKey(0)
    return new_image


#Pathing
DATA_FOLDER = "/media/benjin/Samsung_T5/AMZ/sensor_fusion_data/2020-08-08_alpnach/data/autocross_2020-08-08-13-46-33"
CAMERA = ["forward", "left", "right"][0]
IDX = 100

#Read in data
img_path = os.path.join(DATA_FOLDER, CAMERA + "_camera_filtered",
                        str(IDX).zfill(8) + ".png")
img = cv2.imread(img_path)

di_path = os.path.join(DATA_FOLDER, CAMERA + "_di", str(IDX).zfill(8) + ".bin")
di = np.fromfile(di_path, dtype=np.float16).reshape((640, 2592, 2))

#Draw depth
draw_depth(img, di[:, :, 0], CAMERA)
