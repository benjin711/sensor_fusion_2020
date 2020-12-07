import argparse
import json
import pickle
import sys

from models.experimental import *
from utils.datasets import *


def test(
        data,
        conf_thres,
        iou_thres,
        weights=None,
        batch_size=16,
        imgsz=640,  # for NMS
        save_json=False,
        single_cls=False,
        augment=False,
        verbose=False,
        model=None,
        dataloader=None,
        save_dir='',
        merge=False,
        save_txt=False,
        generate_depth_stats=False,
        stride=32):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        merge, save_txt, save_bin = opt.merge, opt.save_txt, opt.save_bin  # use Merge NMS, save *.txt labels
        if save_txt or save_bin:
            out = Path('inference/output')
            if os.path.exists(out):
                shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder

        # Remove previous
        for f in glob.glob(str(Path(save_dir) / 'test_batch*.jpg')):
            os.remove(f)

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
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

        # For one time run only
        # EVAL_SF_EXTEND_TEST_SET = True
        # if EVAL_SF_EXTEND_TEST_SET:
        #     with open(path, "r") as f:
        #         orig_paths = f.readlines()

        #     with open(os.path.join(os.path.dirname(path), "ben_test_extended.txt"), "w") as f:
        #         cameras = ["forward", "left", "right"]
        #         for orig_path in orig_paths:
        #             which_camera = [camera for camera in cameras if orig_path.find(camera) > 0][0]

        #             for camera in cameras:
        #                 f.write(orig_path.replace(which_camera, camera))
        #                 if not os.path.exists(orig_path.replace(which_camera, camera)):
        #                     tmp = np.zeros((0,3))
        #                     tmp.tofile(orig_path.replace(which_camera, camera).rstrip("\n"))


        dataloader, dataset = \
        create_dataloader(path, imgsz, batch_size, model.stride.max(), opt,
                          hyp=None, augment=False, cache=False, pad=0,
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
    p, r, f1, mp, mr, map50, map, t0, t1, depth_err = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
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

        # Disable gradients
        with torch.no_grad():
            # Run model
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
                                         paths,
                                         img,
                                         conf_thres=conf_thres,
                                         iou_thres=iou_thres,
                                         merge=merge)
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image

        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool),
                                  torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0
                                                  ]]  # normalization gain whwh
                base = os.path.dirname(paths[si])
                base = str(out / Path(base[1:]))
                stem = str(Path(paths[si]).stem) + ".txt"
                os.makedirs(base, exist_ok=True)

                pred[:, :4] = scale_coords(img[si].shape[1:], pred[:, :4],
                                           shapes[si][0],
                                           shapes[si][1])  # to original
                for *xyxy, conf, cls, dpth in pred:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                            gn).view(-1).tolist()  # normalized xywh
                    with open(os.path.join(base, stem), 'a') as f:
                        f.write(('%g ' * 6 + '\n') %
                                (cls, *xywh, dpth))  # label format

            if save_bin:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0
                                                  ]]  # normalization gain whwh
                base = os.path.dirname(paths[si])
                base = str(out / Path(base[1:]))
                stem = str(Path(paths[si]).stem) + ".txt"
                os.makedirs(base, exist_ok=True)

                pred[:, :4] = scale_coords(img[si].shape[1:], pred[:, :4],
                                           shapes[si][0],
                                           shapes[si][1])  # to original

                pred_bin = np.zeros((0, 6))
                for *xyxy, conf, cls, dpth in pred:
                    curr_cone = np.zeros(6)
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                            gn).view(-1).tolist()  # normalized xywh
                    curr_cone[0] = cls
                    curr_cone[1:5] = xywh
                    curr_cone[5] = dpth
                    pred_bin = np.vstack((pred_bin, curr_cone.reshape(1, 6)))

                pred_bin.tofile(
                    os.path.join(base, stem.replace('.txt', '.bin')))

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = paths[si]
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img[si].shape[1:], box, shapes[si][0],
                             shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({
                        'image_id':
                        int(image_id) if image_id.isnumeric() else image_id,
                        'category_id':
                        coco91class[int(p[5])],
                        'bbox': [round(x, 3) for x in b],
                        'score':
                        round(p[4], 5),
                        'depth':
                        p[6]
                    })

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
        # if batch_i < 10:
        #     f = Path(save_dir) / ('test_batch%g_gt.jpg' % batch_i)  # filename
        #     gt_result = plot_images(img, targets, paths, str(f),
        #                             names)  # ground truth
        #     f = Path(save_dir) / ('test_batch%g_pred.jpg' % batch_i)
        #     pred_result = plot_images(img,
        #                               output_to_target(output, width, height),
        #                               paths, str(f), names)  # predictions

    if generate_depth_stats:
        plt.figure()
        ax = plt.gca()
        ax.hist(depth_stats['depth_error'])
        plt.savefig('depth_error_histogram.png', bins=100)
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
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(
            1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64),
                         minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 7  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map,
                np.mean(np.array(depth_stats['depth_error']))))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print("{}, {}, {}, {}, {}, {}, {}".format(names[c], seen, nt[c],
                                                      p[i], r[i], ap50[i],
                                                      ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3
              for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print(
            'Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g'
            % t)

    # Save JSON
    if save_json and len(jdict):
        f = 'detections_val2017_%s_results.json' % \
            (weights.split(os.sep)[-1].replace('.pt', '') if isinstance(weights, str) else '')  # filename
        print('\nCOCO mAP with pycocotools... saving %s...' % f)
        with open(f, 'w') as file:
            json.dump(jdict, file)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
            cocoGt = COCO(
                glob.glob('../coco/annotations/instances_val*.json')
                [0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes(f)  # initialize COCO pred api
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # image IDs to evaluate
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:
                                        2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print('ERROR: pycocotools unable to run: %s' % e)

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, np.mean(np.array(depth_stats['depth_error'])),
            *(loss.cpu() / len(dataloader)).tolist()), maps, t, (gt_result,
                                                                 pred_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights',
                        nargs='+',
                        type=str,
                        default='yolov4mmish.pt',
                        help='model.pt path(s)')
    parser.add_argument('--data',
                        type=str,
                        default='data/amz_tiny.yaml',
                        help='*.data path')
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help='size of each image batch')
    parser.add_argument('--img-size',
                        type=int,
                        default=1280,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.001,
                        help='object confidence threshold')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.65,
                        help='IOU threshold for NMS')
    parser.add_argument('--save-json',
                        action='store_true',
                        help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--save-bin',
                        action='store_true',
                        help='save predictions to bin files')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
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
    parser.add_argument('--save-txt',
                        action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--generate-depth-stats',
                        action='store_true',
                        help='Generate histogram with depth error')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.conf_thres,
             opt.iou_thres,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
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
                               opt.conf_thres, opt.iou_thres, opt.save_json)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.6g')  # save
        os.system('zip -r study.zip study_*.txt')
        # plot_study_txt(f, x)  # plot
