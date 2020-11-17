"""Detect cones in all three images, generate 3D cone positions using
   the camera intrinsics. Fuse 3D cone positions using ICP."""

import argparse
from copy import deepcopy
import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *
from preprocessing.preprocessing.utils.utils import load_camera_calib, load_stereo_calib, load_mrh_transform
from utils import icp2d

def detect(save_img=False):
    out, forward_label_fp, weights, view_img, save_txt, imgsz, rect = \
        opt.output, opt.forward_paths, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.rect

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    save_img = True
    original_shape = (640, 2592)
    dataset = LoadJointImages(forward_label_fp, img_size=imgsz, rect=rect)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors = [[0, 0, 255], [255, 255, 0]]

    # Load camera calibration
    calibs = {'forward': None,
              'left': None,
              'right': None}
    forward_hfov = 40

    # Run inference
    # img = torch.zeros((1, 5, imgsz, imgsz), device=device)  # init img
    # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for (path_f, path_l, path_r), (img_f, img_l, img_r), (img_f0, img_l0, img_r0) in dataset:

        img_f = torch.from_numpy(img_f).to(device)
        img_f = img_f.half() if half else img_f.float()  # uint8 to fp16/32
        img_f[:3, :, :] /= 255.0  # rescale RGB channels so that 0 - 255 to 0.0 - 1.0
        img_f[3, :, :] /= 255.0 # Rescale depth channel. Max depth found was 202 m
        if img_f.ndimension() == 3:
            img_f = img_f.unsqueeze(0)

        img_l = torch.from_numpy(img_l).to(device)
        img_l = img_l.half() if half else img_l.float()  # uint8 to fp16/32
        img_l[:3, :,
        :] /= 255.0  # rescale RGB channels so that 0 - 255 to 0.0 - 1.0
        img_l[3, :,
        :] /= 255.0  # Rescale depth channel. Max depth found was 202 m
        if img_l.ndimension() == 3:
            img_l = img_l.unsqueeze(0)

        img_r = torch.from_numpy(img_r).to(device)
        img_r = img_r.half() if half else img_r.float()  # uint8 to fp16/32
        img_r[:3, :,
        :] /= 255.0  # rescale RGB channels so that 0 - 255 to 0.0 - 1.0
        img_r[3, :,
        :] /= 255.0  # Rescale depth channel. Max depth found was 202 m
        if img_r.ndimension() == 3:
            img_r = img_r.unsqueeze(0)

        # Inference
        pred_f = model(img_f)[0]
        pred_l = model(img_l)[0]
        pred_r = model(img_r)[0]

        # Apply NMS
        pred_f = non_max_suppression(pred_f, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        pred_l = non_max_suppression(pred_l, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        pred_r = non_max_suppression(pred_r, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Load transformation from each camera to the MRH frame
        img_dir, _ = os.path.split(path_f)
        calibs['forward'] = load_camera_calib(os.path.join(img_dir, '..', '..', '..', 'static_transformations', 'forward.yaml'))
        calibs['forward']['T'] = load_mrh_transform(os.path.join(img_dir, '..', '..', '..', 'static_transformations', 'extrinsics_mrh_forward.yaml'), invert=True)
        calibs['left'] = load_camera_calib(os.path.join(img_dir, '..', '..', '..', 'static_transformations', 'left.yaml'))
        calibs['left']['T'] = load_mrh_transform(os.path.join(img_dir, '..', '..', '..', 'static_transformations', 'extrinsics_mrh_left.yaml'), invert=True)
        calibs['right'] = load_camera_calib(os.path.join(img_dir, '..', '..', '..', 'static_transformations', 'right.yaml'))
        calibs['right']['T'] = load_mrh_transform(os.path.join(img_dir, '..', '..', '..', 'static_transformations', 'extrinsics_mrh_right.yaml'), invert=True)

        # DEBUG : Load original images for visualizing boxes
        img_f_raw = cv2.imread(path_f)
        img_l_raw = cv2.imread(path_l)
        img_r_raw = cv2.imread(path_r)

        # Process detections
        xyz_f, xyz_l, xyz_r = None, None, None
        for i, det in enumerate(pred_f):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_f.shape[2:], det[:, :4],
                                          img_f_raw.shape[:2]).round()

            xy_tip = deepcopy(det[:, :2])
            xy_tip[:, 0] = 0.5*(det[:, 0] + det[:, 2])

            u_tip = (xy_tip[:, 0] - calibs['forward']['camera_matrix'][0, 2]).numpy()
            tx_tip = np.arctan(u_tip/calibs['forward']['camera_matrix'][0, 0])
            v_tip = (xy_tip[:, 1] - calibs['forward']['camera_matrix'][1, 2]).numpy()
            ty_tip = np.arctan(v_tip/calibs['forward']['camera_matrix'][1, 1])

            depth_tip = det[:, -1].numpy()
            x_tip = np.multiply(depth_tip, np.sin(tx_tip)).reshape((-1, 1))
            z_tip_x = np.multiply(depth_tip, np.cos(tx_tip))
            y_tip = np.multiply(depth_tip, np.sin(ty_tip)).reshape((-1, 1))
            z_tip_y = np.multiply(depth_tip, np.cos(ty_tip))
            z_tip = 0.5*(z_tip_x + z_tip_y).reshape((-1, 1))
            # z_tip = z_tip_x.reshape((-1, 1))
            xyz_f = np.concatenate([x_tip, y_tip, z_tip], axis=1)

            # Draw cones
            for *xyxy, conf, cls, depth in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, img_f_raw, label=label, color=colors[int(cls)], line_thickness=2,
                             depth=depth)

        for i, det in enumerate(pred_l):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_l.shape[2:], det[:, :4],
                                          img_l_raw.shape[:2]).round()

            xy_tip = deepcopy(det[:, :2])
            xy_tip[:, 0] = 0.5 * (det[:, 0] + det[:, 2])

            u_tip = (xy_tip[:, 0] - calibs['left']['camera_matrix'][
                0, 2]).numpy()
            tx_tip = np.arctan(u_tip / calibs['left']['camera_matrix'][0, 0])
            v_tip = (xy_tip[:, 1] - calibs['left']['camera_matrix'][
                1, 2]).numpy()
            ty_tip = np.arctan(v_tip / calibs['left']['camera_matrix'][1, 1])

            depth_tip = det[:, -1].numpy()
            x_tip = np.multiply(depth_tip, np.sin(tx_tip)).reshape((-1, 1))
            z_tip_x = np.multiply(depth_tip, np.cos(tx_tip))
            y_tip = np.multiply(depth_tip, np.sin(ty_tip)).reshape((-1, 1))
            z_tip_y = np.multiply(depth_tip, np.cos(ty_tip))
            z_tip = 0.5*(z_tip_x + z_tip_y).reshape((-1, 1))
            # z_tip = z_tip_x.reshape((-1, 1))
            xyz_l = np.concatenate([x_tip, y_tip, z_tip], axis=1)

            for *xyxy, conf, cls, depth in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, img_l_raw, label=label, color=colors[int(cls)], line_thickness=2,
                             depth=depth)

        for i, det in enumerate(pred_r):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_r.shape[2:], det[:, :4],
                                          img_r_raw.shape[:2]).round()

            xy_tip = deepcopy(det[:, :2])
            xy_tip[:, 0] = 0.5 * (det[:, 0] + det[:, 2])

            u_tip = (xy_tip[:, 0] - calibs['right']['camera_matrix'][
                0, 2]).numpy()
            tx_tip = np.arctan(u_tip / calibs['right']['camera_matrix'][0, 0])
            v_tip = (xy_tip[:, 1] - calibs['right']['camera_matrix'][
                1, 2]).numpy()
            ty_tip = np.arctan(v_tip / calibs['right']['camera_matrix'][1, 1])

            depth_tip = det[:, -1].numpy()
            x_tip = np.multiply(depth_tip, np.sin(tx_tip)).reshape((-1, 1))
            z_tip_x = np.multiply(depth_tip, np.cos(tx_tip))
            y_tip = np.multiply(depth_tip, np.sin(ty_tip)).reshape((-1, 1))
            z_tip_y = np.multiply(depth_tip, np.cos(ty_tip))
            z_tip = 0.5*(z_tip_x + z_tip_y).reshape((-1, 1))
            # z_tip = z_tip_x.reshape((-1, 1))
            xyz_r = np.concatenate([x_tip, y_tip, z_tip], axis=1)

            for *xyxy, conf, cls, depth in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, img_r_raw, label=label,
                             color=colors[int(cls)], line_thickness=2,
                             depth=depth)

        img_concat_det = np.concatenate([img_l_raw, img_f_raw, img_r_raw], axis=0)
        img_concat_det = cv2.resize(img_concat_det, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Detections", img_concat_det)
        cv2.waitKey(0)

        # Transform cones to forward camera frame
        xyz_f_homo = np.concatenate([xyz_f, np.ones((xyz_f.shape[0], 1))], axis=1)
        xyz_l_homo = np.concatenate([xyz_l, np.ones((xyz_l.shape[0], 1))],
                                    axis=1)
        xyz_r_homo = np.concatenate([xyz_r, np.ones((xyz_r.shape[0], 1))],
                                    axis=1)
        xyz_f_homo = (calibs['forward']['T'] @ xyz_f_homo.T).T
        xyz_l_homo = (calibs['left']['T'] @ xyz_l_homo.T).T
        xyz_r_homo = (calibs['right']['T'] @ xyz_r_homo.T).T
        xyz_f = xyz_f_homo[:, :3]
        xyz_l = xyz_l_homo[:, :3]
        xyz_r = xyz_r_homo[:, :3]

        plt.figure()
        plt.scatter(-xyz_l[:, 0], -xyz_l[:, 1])
        plt.scatter(-xyz_f[:, 0], -xyz_f[:, 1])
        plt.scatter(-xyz_r[:, 0], -xyz_r[:, 1])
        plt.title('Cones')
        plt.xlabel('X (m)')
        plt.ylabel('-Y (m)')
        plt.legend(["Left", "Forward", "Right"])
        plt.show()

        # # Only use points within +/- 80 deg for correspondence
        # angle_l = np.rad2deg(np.arctan2(xyz_l[:, 0], xyz_l[:, 2]))
        # mask_l = np.logical_and(angle_l < forward_hfov, angle_l > -forward_hfov)
        # mask_l = np.logical_and(mask_l, xyz_l[:, 2] < 30)
        # angle_r = np.rad2deg(np.arctan2(xyz_r[:, 0], xyz_r[:, 2]))
        # mask_r = np.logical_and(angle_r < forward_hfov, angle_r > -forward_hfov)
        # mask_r = np.logical_and(mask_r, xyz_r[:, 2] < 30)
        # T_l = icp2d.icp(xyz_f[:, [0, 2]], xyz_l[:, [0, 2]][mask_l])
        # T_r = icp2d.icp(xyz_f[:, [0, 2]], xyz_r[:, [0, 2]][mask_r])
        #
        # xyz_l[:, [0, 2]] = (T_l @ np.concatenate([xyz_l[:, [0, 2]], np.zeros((xyz_l.shape[0], 1))], axis=1).T).T
        # xyz_r[:, [0, 2]] = (T_r @ np.concatenate(
        #     [xyz_r[:, [0, 2]], np.zeros((xyz_r.shape[0], 1))], axis=1).T).T
        #
        # plt.figure()
        # plt.scatter(xyz_l[:, 0], xyz_l[:, 2])
        # plt.scatter(xyz_f[:, 0], xyz_f[:, 2])
        # plt.scatter(xyz_r[:, 0], xyz_r[:, 2])
        # plt.xlabel('X (m)')
        # plt.ylabel('Z (m)')
        # plt.legend(["Left", "Forward", "Right"])
        # plt.title("Merged Cones")
        # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4l-mish.pt', help='model.pt path(s)')
    parser.add_argument('--forward-paths', type=str, help='Path to text file, containing paths to forward camera labels.')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.05, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--rect', action='store_true', help='rectangular images')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                detect()
                create_pretrained(opt.weights, opt.weights)
        else:
            detect()
