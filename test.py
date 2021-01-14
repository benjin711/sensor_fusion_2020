import argparse
import json
import pickle
import sys
import torch
import numpy as np
import yaml
import torch

from models.experimental import *
from utils.datasets import *

# Hyperparameters
hyp = {
    'optimizer': 'SGD',  # ['adam', 'SGD', None] if none, default is SGD
    'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
    'momentum': 0.937,  # SGD momentum/Adam beta1
    'weight_decay': 5e-4,  # optimizer weight decay
    'depth': 1e-2,  # depth loss gain TODO: tune it, or learn it
    'giou': 0.05,  # giou loss gain
    'cls': 5,  # cls loss gain
    'cls_pw': 1.0,  # cls BCELoss positive_weight
    'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
    'obj_pw': 1.0,  # obj BCELoss positive_weight
    'iou_t': 0.20,  # iou training threshold
    'anchor_t': 4.0,  # anchor-multiple threshold
    'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
    'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
    'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
    'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
    'degrees': 0.0,  # image rotation (+/- deg)
    'translate': 0.0,  # image translation (+/- fraction)
    'scale': 0.5,  # image scale (+/- gain)
    'shear': 0.0
}  # image shear (+/- deg)


def test(
        data,
        conf_thres,
        iou_thres,
        weights=None,
        batch_size=16,
        imgsz=640,  # for NMS
        single_cls=False,
        augment=False,
        verbose=False,
        model=None,
        dataloader=None,
        save_dir='',
        merge=False,
        save_pkl=True,
        generate_depth_stats=False,
        stride=32):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        merge, save_pkl = opt.merge, opt.save_pkl  # use Merge NMS, save *.txt labels

        # Remove previous
        for f in glob.glob(str(Path(save_dir) / 'test_batch*.jpg')):
            os.remove(f)

        # Load model
        model, model_info = attempt_load(
            weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95,
                          10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        path = data['test'] if opt.task == 'test' else data[
            'val']  # path to val/test images

        dataloader, dataset = \
        create_dataloader(path, imgsz, batch_size, model.stride.max(), opt,
                          hyp=hyp, augment=False, cache=False, pad=0,
                          rect=True)
        tmp_img = dataset[0][0]
        img_shape = np.array(tmp_img.detach().cpu().numpy().shape[1:])
        img = torch.zeros((1, 5, img_shape[0], img_shape[1]),
                          device=device)  # init img
        _ = model(img.half() if half else img
                  ) if device.type != 'cpu' else None  # run once

    seen = 0
    names = model.names if hasattr(model, 'names') else model.module.names
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 7) % ('Class', 'Images', 'Targets', 'P', 'R',
                                 'mAP@.5', 'mAP@.5:.95', 'Depth Error')
    p, r, f1, mp, mr, map50, map75, map, t0, t1, depth_err = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(4, device=device)
    total_boxes = torch.zeros(1, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    depth_stats = {
        'box_width': [],
        'box_height': [],
        'depth_true': [],
        'depth_pred': [],
        'depth_error': []
    }

    # To save predictions as pkl file
    pred_pkl = {}

    for batch_i, (img, targets, paths,
                  shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img[:, :3, :, :] /= 255.0  # Rescale RGB
        img[:, 3, :, :] /= 255.0  # Rescale depth
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)
        total_boxes += targets.shape[0]

        if opt.rgb_drop:
            img[:, :3, :, :] *= torch.randint(0, 2,
                                              size=(1, )).to(device,
                                                             non_blocking=True)

        # Disable gradients
        with torch.no_grad():

            ### RUN MODEL ###
            t = torch_utils.time_synchronized()
            inf_out, train_out = model(
                img, augment=augment)  # inference and training outputs
            t0 += torch_utils.time_synchronized() - t

            # Compute loss
            if training:  # if model has loss hyperparameters
                tmp, loss_items = compute_loss([x.float() for x in train_out],
                                               targets,
                                               model)  # GIoU, obj, cls, depth
                loss += loss_items[:4]

            # Run NMS
            t = torch_utils.time_synchronized()
            output = non_max_suppression(inf_out,
                                         conf_thres=conf_thres,
                                         iou_thres=iou_thres,
                                         merge=merge)
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        for si, (pred, path) in enumerate(zip(output, paths)):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool),
                                  torch.Tensor(), torch.Tensor(), tcls))
                continue

            if save_pkl:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]].to(
                    device)  # normalization gain whwh

                data_base = "sensor_fusion_data"
                index = path.find(data_base)
                index += len(data_base) + 1

                pred_tmp = scale_coords(img[si].shape[1:], pred[:, :4],
                                        shapes[si][0],
                                        shapes[si][1])  # to original

                pred_pkl_ = np.zeros((0, 7))
                for *xyxy, (conf, cls, dpth) in zip(pred_tmp, pred[:, 4:]):
                    curr_cone = np.zeros(7)
                    xywh = (xyxy2xywh(xyxy[0].view(1, 4)) /
                            gn).view(-1).tolist()  # normalized xywh
                    curr_cone[0] = cls
                    curr_cone[1:5] = xywh
                    curr_cone[5] = dpth
                    curr_cone[6] = conf
                    pred_pkl_ = np.vstack((pred_pkl_, curr_cone.reshape(1, 7)))

                pred_pkl[path[index:]] = pred_pkl_

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0],
                                  niou,
                                  dtype=torch.bool,
                                  device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]
                tdepth_tensor = labels[:, 1]

                # target boxes
                tbox = xywh2xyxy(labels[:, 2:6]) * whwh

                pred_depth = pred[:, 6]
                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(
                        -1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero().view(
                        -1)  # prediction indices
                    pdepth = pred_depth[pi]

                    # Search for detections of that class
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(
                            1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                depth_gt = float(tdepth_tensor[d])
                                depth = float(pdepth[j])
                                box = pred[j, :4]
                                box_h, box_w = float(box[0, 3] -
                                                     box[0,
                                                         1]), float(box[0, 2] -
                                                                    box[0, 0])

                                # Log depth statistics
                                depth_stats['box_width'].append(box_w)
                                depth_stats['box_height'].append(box_h)
                                depth_stats['depth_pred'].append(depth)
                                depth_stats['depth_true'].append(depth_gt)
                                depth_stats['depth_error'].append(
                                    abs(depth - depth_gt))

                                detected.append(d)
                                correct[
                                    pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(
                                        detected
                                ) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append(
                (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            # depth_stats.append(pred[:, 6])

        # Plot images
        # [xywh, depth, obj_prob, class_1_prob, class_2_prob, ... class_nc_prob]
        if batch_i < 1:
            f = Path(save_dir) / ('test_batch%g_gt.jpg' % batch_i)  # filename
            gt_result = plot_images(img, targets, paths, str(f),
                                    names)  # ground truth
            f = Path(save_dir) / ('test_batch%g_pred.jpg' % batch_i)
            pred_result = plot_images(img,
                                      output_to_target(output, width, height),
                                      paths, str(f), names)  # predictions

    if save_pkl:
        with open('inference.pkl', 'wb') as inference_f:
            pickle.dump(pred_pkl, inference_f, pickle.HIGHEST_PROTOCOL)

    if generate_depth_stats:
        plt.figure()
        ax = plt.gca()
        ax.hist(depth_stats['depth_error'], bins=50)
        ax.set_xlabel("Depth Error")
        ax.set_ylabel("Number of Predictions")
        ax.set_title("Depth Error Histogram")
        plt.savefig('depth_error_histogram.png')
        plt.show()
        print(
            f"Median error: {np.median(np.array(depth_stats['depth_error']))}")
        print(f"Std error: {np.std(np.array(depth_stats['depth_error']))}")

        with open('depth_stats.pkl', 'wb') as depth_stats_f:
            pickle.dump(depth_stats, depth_stats_f, pickle.HIGHEST_PROTOCOL)

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap75, ap = p[:, 0], r[:, 0], ap[:, 0], ap[:, 5], ap.mean(
            1)  # [P, R, AP@0.5, AP@0.75, AP@0.5:0.95]
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(
        ), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64),
                         minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    print("_, num imgs, num targets, mp, mr, map50, map75, map, mdeptherror")
    pf = '%20s' + '%12.3g' * 8  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map75, map,
                np.mean(np.array(depth_stats['depth_error']))))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print("{}, {}, {}, {}, {}, {}, {}, {}".format(
                names[c], seen, nt[c], p[i], r[i], ap50[i], ap75[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (
        height, width, batch_size)  # tuple
    if not training:
        print(
            'Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g'
            % t)

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    # Model Summary
    # Testset Size,
    # Class Yellow, Class Blue
    # Number Targets Blue, Number Targets Yellow, Number Targets Total,
    # Precision Yellow, Precision Blue, Precision Mean,
    # Recall Yellow, Recall Blue, Recall Mean,
    # ap50, ap75, ap Yellow, ap50, ap75, ap Blue, map50, map75, map
    # mdeptherror
    # Image height, width
    # Inference speed at batch size

    metrics = {
        "Num Layers":
        model_info[0],
        "Num Parameters":
        model_info[1],
        "Num Gradients":
        model_info[2],
        "Yellow Class ID":
        names.index('yellow'),
        "Blue Class ID":
        names.index('blue'),
        "Num Blue Targets":
        int(nt[names.index('blue')]),
        "Num Yellow Targets":
        int(nt[names.index('yellow')]),
        "Num Targets Total":
        int(sum(nt)),
        "Yellow Precision":
        float(p[names.index('yellow')]),
        "Blue Precision":
        float(p[names.index('blue')]),
        "Mean Precision":
        float(p.mean()),
        "Yellow Recall":
        float(r[names.index('yellow')]),
        "Blue Recall":
        float(r[names.index('blue')]),
        "Mean Recall":
        float(r.mean()),
        "Yellow AP50":
        float(ap50[names.index('yellow')]),
        "Yellow AP75":
        float(ap75[names.index('yellow')]),
        "Yellow AP50:95":
        float(ap[names.index('yellow')]),
        "Blue AP50":
        float(ap50[names.index('blue')]),
        "Blue AP75":
        float(ap75[names.index('blue')]),
        "Blue AP50:95":
        float(ap[names.index('blue')]),
        "mAP50":
        float(ap50.mean()),
        "mAP75":
        float(ap75.mean()),
        "mAP50:95":
        float(ap.mean()),
        "Mean Depth Error in m":
        float(np.mean(np.array(depth_stats['depth_error']))),
        "Image Height":
        height,
        "Image Width":
        width,
        "Model Inference Speed in ms per Image":
        t[0],
        "NMS and Filtering Speed in ms per Image":
        t[1],
        "Total Inference Speed in ms per Image":
        t[2],
        "Batch Size":
        batch_size
    }

    with open("metrics.yaml", 'w') as f:
        f.write('# Model Info & Evaluation Metrics: \n\n')
        yaml.dump(metrics, f, sort_keys=False)

    return (mp, mr, map50, map, np.mean(np.array(depth_stats['depth_error'])),
            *(loss.cpu() / len(dataloader)).tolist()), maps, t, (gt_result,
                                                                 pred_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights',
                        nargs='+',
                        type=str,
                        default='weights/best_exp06.pt',
                        help='model.pt path(s)')
    parser.add_argument('--data',
                        type=str,
                        default='data/amz_data_splits.yaml',
                        help='*.data path')
    parser.add_argument('--batch-size',
                        type=int,
                        default=1,
                        help='size of each image batch')
    parser.add_argument('--img-size',
                        type=int,
                        default=1280,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.01,
                        help='object confidence threshold')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.3,
                        help='IOU threshold for NMS')
    parser.add_argument('--task',
                        default='test',
                        help="'val', 'test', 'study'")
    parser.add_argument('--device',
                        default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls',
                        action='store_true',
                        help='treat as single-class dataset')
    parser.add_argument('--augment',
                        action='store_true',
                        help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='report mAP by class')
    parser.add_argument('--save-pkl',
                        action='store_true',
                        help='save predictions to *.pkl')
    parser.add_argument(
        '--rgb_drop',
        action='store_true',
        help='Inference on only lidar depth layer for 50% of the time')
    parser.add_argument('--generate-depth-stats',
                        action='store_true',
                        help='Generate histogram with depth error')
    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ['val', 'test']:  # run normally
        ret = test(data=opt.data,
                   weights=opt.weights,
                   batch_size=opt.batch_size,
                   imgsz=opt.img_size,
                   conf_thres=opt.conf_thres,
                   iou_thres=opt.iou_thres,
                   single_cls=opt.single_cls,
                   augment=opt.augment,
                   verbose=opt.verbose,
                   generate_depth_stats=opt.generate_depth_stats)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in [
                'yolov4s-mish.pt', 'yolov4m-mish.pt', 'yolov4l-mish.pt',
                'yolov4x-mish.pt'
        ]:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem
                                     )  # filename to save to
            x = list(range(288, 896, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i,
                               opt.conf_thres, opt.iou_thres)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.6g')  # save
        os.system('zip -r study.zip study_*.txt')
        # plot_study_txt(f, x)  # plot
