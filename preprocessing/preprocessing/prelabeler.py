import torch
import cv2 as cv

import os

from utils.models import Darknet
import utils.utils as utils
import utils.model_utils as model_utils


def prelabeler(cfg):
    """Load model from the cfg. Configure the image paths.
    Configure the camera calibration. Undistort, forward pass the images,
    Generate Supervisely labels for them in JSON format.

    Expected keys inside the cfg: ['model_config', 'model_weights', 'camera_config', 'image_directory']
    where the values are the paths to these items."""

    # Get camera calibration parameters
    camera_calib = utils.load_camera_calib(cfg["camera_config"])

    # Load YOLO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(cfg["model_config"]).to(device)
    if cfg["model_weights"].endswith(".weights"):
        model.load_darknet_weights(model["model_weights"])
    else:
        model.load_state_dict(
            torch.load(model["model_weights"]))
    model.eval()
    model.to(device)
    conf_thresh, nms_thresh = 0.3, 0.10

    # Iterate through the image directory and forward pass
    image_paths = os.listdir(cfg["image_directory"])

    for image_path in image_paths:
        image_path = os.path.join(cfg["image_directory"], image_path)
        raw_image = cv.imread(image_path)
        undistorted_image = cv.undistort(raw_image, camera_calib["camera_matrix"], camera_calib["distortion_coefficients"])
        tensor_model, img_model = utils.model_utils.process_img(undistorted_image, output_size=(model.img_w, model.img_h))

        # Run inference and postprocess
        output_model = model(tensor_model)
        output_model = model_utils.non_max_suppression(output_model, conf_thresh,
                                                       nms_thresh)
        boxes, labels, configs = output_model[0][:, :4], output_model[0][:,-1], output_model[0][:, 4]

if __name__ == "__main__":
    cfg = {}
    cfg["camera_config"] = '../resources/forward.yaml'
    cfg["model_config"] = '../resources/yolov3-tiny-amz.cfg'
    cfg["model_weights"] = '../resources/yolov3-tiny-amz.weights'
    cfg["image_directory"] = ""
