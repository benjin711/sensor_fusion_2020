import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch.nn.functional as F
import torch
import albumentations as A
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.utils import xyxy2xywh, xywh2xyxy, torch_distributed_zero_first

help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(path,
                      imgsz,
                      batch_size,
                      stride,
                      opt,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      local_rank=-1,
                      world_size=1,
                      num_samples=0):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache.
    with torch_distributed_zero_first(local_rank):
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augment images
            hyp=hyp,  # augmentation hyperparameters
            rect=rect,  # rectangular training
            cache_images=cache,
            single_cls=opt.single_cls,
            stride=int(stride),
            pad=pad,
            num_samples=num_samples)

    batch_size = min(batch_size, len(dataset))
    nw = min(
        [os.cpu_count() // world_size, batch_size if batch_size > 1 else 0,
         4])  # number of workers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset) if local_rank != -1 else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class LoadJointImages:
    def __init__(self, path, img_size=640, rect=False, stride=32):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        """Expects path to a txt file containing paths to forward camera
        labels. Image paths for forward, left, and right are generated from 
        those."""

        with open(path, 'r') as label_f:
            labels = label_f.read().split('\n')
            labels.pop()  # \n on the last line results in empty list element

        self.forward_images = [
            label_path.replace('labels',
                               'camera_filtered').replace('.txt', '.png')
            for label_path in labels
        ]
        self.left_images = [
            forward_path.replace('forward', 'left')
            for forward_path in self.forward_images
        ]
        self.right_images = [
            forward_path.replace('forward', 'right')
            for forward_path in self.forward_images
        ]

        self.forward_dis = [
            image_path.replace('camera_filtered',
                               'di').replace('.png', '.bin')
            for image_path in self.forward_images
        ]
        self.forward_masks = [
            image_path.replace('camera_filtered', 'm').replace('.png', '.bin')
            for image_path in self.forward_images
        ]

        self.left_dis = [
            image_path.replace('camera_filtered',
                               'di').replace('.png', '.bin')
            for image_path in self.left_images
        ]
        self.left_masks = [
            image_path.replace('camera_filtered', 'm').replace('.png', '.bin')
            for image_path in self.left_images
        ]

        self.right_dis = [
            image_path.replace('camera_filtered',
                               'di').replace('.png', '.bin')
            for image_path in self.right_images
        ]
        self.right_masks = [
            image_path.replace('camera_filtered', 'm').replace('.png', '.bin')
            for image_path in self.right_images
        ]

        ni = len(self.forward_images)
        # ni, nv = len(images), len(videos)

        # Assumes width dimension longer than height
        self.rect = rect
        raw_shape = np.array(cv2.imread(self.forward_images[0]).shape[:2])
        if self.rect:
            normalized_shape = raw_shape / raw_shape[1]
            self.img_shape = np.ceil(
                normalized_shape * img_size / stride).astype(np.int) * stride
        else:
            self.img_shape = np.array([img_size, img_size])

        self.img_size = img_size
        self.nf = ni  # number of files
        self.video_flag = False
        self.mode = 'images'

        assert self.nf > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (p, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration

        forward_im_path = self.forward_images[self.count]
        forward_di_path = self.forward_dis[self.count]
        forward_m_path = self.forward_masks[self.count]

        left_im_path = self.left_images[self.count]
        left_di_path = self.left_dis[self.count]
        left_m_path = self.left_masks[self.count]

        right_im_path = self.right_images[self.count]
        right_di_path = self.right_dis[self.count]
        right_m_path = self.right_masks[self.count]

        # Read image
        self.count += 1
        img_f = cv2.imread(forward_im_path)  # BGR
        di_f = np.fromfile(forward_di_path,
                           dtype=np.float16).reshape(img_f.shape[0],
                                                     img_f.shape[1], 2)
        d_f = di_f[:, :, 0].reshape(img_f.shape[0], img_f.shape[1], 1)
        i_f = di_f[:, :, 1].reshape(img_f.shape[0], img_f.shape[1], 1)
        m_f = np.fromfile(forward_m_path,
                          dtype=np.bool).reshape(img_f.shape[0],
                                                 img_f.shape[1], 1)

        img_l = cv2.imread(left_im_path)  # BGR
        di_l = np.fromfile(left_di_path,
                           dtype=np.float16).reshape(img_l.shape[0],
                                                     img_l.shape[1], 2)
        d_l = di_l[:, :, 0].reshape(img_l.shape[0], img_l.shape[1], 1)
        i_l = di_l[:, :, 1].reshape(img_l.shape[0], img_l.shape[1], 1)
        m_l = np.fromfile(left_m_path,
                          dtype=np.bool).reshape(img_l.shape[0],
                                                 img_l.shape[1], 1)

        img_r = cv2.imread(right_im_path)  # BGR
        di_r = np.fromfile(right_di_path,
                           dtype=np.float16).reshape(img_r.shape[0],
                                                     img_r.shape[1], 2)
        d_r = di_r[:, :, 0].reshape(img_r.shape[0], img_r.shape[1], 1)
        i_r = di_r[:, :, 1].reshape(img_r.shape[0], img_r.shape[1], 1)
        m_r = np.fromfile(right_m_path,
                          dtype=np.bool).reshape(img_r.shape[0],
                                                 img_r.shape[1], 1)
        assert img_f is not None, 'Forward image Not Found ' + forward_im_path
        assert img_l is not None, 'Left image Not Found ' + left_im_path
        assert img_r is not None, 'Right image Not Found ' + right_im_path
        # print('image %g/%g %s: ' % (self.count, self.nf, im_path), end='')

        img_f = img_f[:, :, ::-1]
        img_f0 = np.concatenate((img_f, d_f, m_f), axis=-1)
        img_l = img_l[:, :, ::-1]
        img_l0 = np.concatenate((img_l, d_f, m_f), axis=-1)
        img_r = img_r[:, :, ::-1]
        img_r0 = np.concatenate((img_r, d_f, m_f), axis=-1)

        # Padded resize
        img_f = letterbox(img_f0, new_shape=self.img_shape)[0]
        img_f = img_f.transpose(2, 0, 1)
        img_l = letterbox(img_l0, new_shape=self.img_shape)[0]
        img_l = img_l.transpose(2, 0, 1)
        img_r = letterbox(img_r0, new_shape=self.img_shape)[0]
        img_r = img_r.transpose(2, 0, 1)

        # Convert
        img_f = np.ascontiguousarray(img_f)
        img_l = np.ascontiguousarray(img_l)
        img_r = np.ascontiguousarray(img_r)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return (forward_im_path, left_im_path,
                right_im_path), (img_f, img_l, img_r), (img_f0, img_l0, img_r0)

    def __len__(self):
        return self.nf  # number of files


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, rect=False, stride=32):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        """Expects path to a directory with folders named *_filtered
        *_di, and *_m. Where *_filtered contains .png images, *_di
        and *_m contains .bin files."""

        self.images = glob.glob(os.path.join(p, '*_camera_filtered', '*.png'))
        self.dis = [
            x.replace('camera_filtered', 'di').replace('.png', '.bin')
            for x in self.images
        ]
        self.masks = [
            x.replace('camera_filtered', 'm').replace('.png', '.bin')
            for x in self.images
        ]
        # if '*' in p:
        #     files = sorted(glob.glob(p))  # glob
        # elif os.path.isdir(p):
        #     files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        # elif os.path.isfile(p):
        #     files = [p]  # files
        # else:
        #     raise Exception('ERROR: %s does not exist' % p)

        ni = len(self.images)
        # ni, nv = len(images), len(videos)

        # Assumes width dimension longer than height
        self.rect = rect
        raw_shape = np.array(cv2.imread(self.images[0]).shape[:2])
        if self.rect:
            normalized_shape = raw_shape / raw_shape[1]
            self.img_shape = np.ceil(
                normalized_shape * img_size / stride).astype(np.int) * stride
        else:
            self.img_shape = np.array([img_size, img_size])

        self.img_size = img_size
        self.files = self.images
        self.nf = ni  # number of files
        self.video_flag = False
        self.mode = 'images'

        assert self.nf > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (p, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        im_path = self.images[self.count]
        di_path = self.dis[self.count]
        m_path = self.masks[self.count]
        self.cap = None
        # if self.video_flag[self.count]:
        #     # Read video
        #     self.mode = 'video'
        #     ret_val, img0 = self.cap.read()
        #     if not ret_val:
        #         self.count += 1
        #         self.cap.release()
        #         if self.count == self.nf:  # last video
        #             raise StopIteration
        #         else:
        #             path = self.files[self.count]
        #             self.new_video(path)
        #             ret_val, img0 = self.cap.read()
        #
        #     self.frame += 1
        #     print('video %g/%g (%g/%g) %s: ' %
        #           (self.count + 1, self.nf, self.frame, self.nframes, path),
        #           end='')

        # Read image
        self.count += 1
        img0 = cv2.imread(im_path)  # BGR
        di0 = np.fromfile(di_path,
                          dtype=np.float16).reshape(img0.shape[0],
                                                    img0.shape[1], 2)
        d0 = di0[:, :, 0].reshape(img0.shape[0], img0.shape[1], 1)
        i0 = di0[:, :, 1].reshape(img0.shape[0], img0.shape[1], 1)
        m0 = np.fromfile(m_path,
                         dtype=np.bool).reshape(img0.shape[0], img0.shape[1],
                                                1)

        assert img0 is not None, 'Image Not Found ' + im_path
        print('image %g/%g %s: ' % (self.count, self.nf, im_path), end='')

        img0 = img0[:, :, ::-1]
        img0 = np.concatenate((img0, d0, m0), axis=-1)
        # Padded resize
        img = letterbox(img0, new_shape=self.img_shape)[0]
        img = img.transpose(2, 0, 1)
        # Convert
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return im_path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe=0, img_size=640):
        self.img_size = img_size

        if pipe == '0':
            pipe = 0  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa'  # IP traffic camera
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        # https://answers.opencv.org/question/215996/changing-gstreamer-pipeline-to-opencv-in-pythonsolved/
        # pipe = '"rtspsrc location="rtsp://username:password@192.168.1.64/1" latency=10 ! appsink'  # GStreamer

        # https://answers.opencv.org/question/200787/video-acceleration-gstremer-pipeline-in-videocapture/
        # https://stackoverflow.com/questions/54095699/install-gstreamer-support-for-opencv-python-package  # install help
        # pipe = "rtspsrc location=rtsp://root:root@192.168.0.91:554/axis-media/media.amp?videocodec=h264&resolution=3840x2160 protocols=GST_RTSP_LOWER_TRANS_TCP ! rtph264depay ! queue ! vaapih264dec ! videoconvert ! appsink"  # GStreamer

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, 'Camera Error %s' % self.pipe
        img_path = 'webcam.jpg'
        print('webcam %g: ' % self.count, end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640):
        self.mode = 'images'
        self.img_size = img_size

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [
                    x.strip() for x in f.read().splitlines() if len(x.strip())
                ]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            cap = cv2.VideoCapture(0 if s == '0' else s)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([
            letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs
        ], 0)  # inference shapes
        self.rect = np.unique(
            s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print(
                'WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.'
            )

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [
            letterbox(x, new_shape=self.img_size, auto=self.rect)[0]
            for x in img0
        ]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1,
                                           2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 num_samples=0,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0):
        try:
            """Fetch the label paths from the top-level directory given"""
            with open(path, 'r') as label_f:
                self.label_files = label_f.read().split('\n')
                self.label_files.pop()  # Remove empty line

            self.img_files = [
                x.replace('labels', 'camera_filtered').replace('.txt', '.png')
                for x in self.label_files
            ]
            self.di_files = [
                x.replace('labels', 'di').replace('.txt', '.bin')
                for x in self.label_files
            ]
            self.m_files = [
                x.replace('labels', 'm').replace('.txt', '.bin')
                for x in self.label_files
            ]
            # f = [] # image files
            # for p in path if isinstance(path, list) else [path]:
            #     p = str(Path(p))  # os-agnostic
            #     parent = str(Path(p).parent) + os.sep
            #     if os.path.isfile(p):  # file
            #         with open(p, 'r') as t:
            #             t = t.read().splitlines()
            #             f += [
            #                 x.replace('./', parent)
            #                 if x.startswith('./') else x for x in t
            #             ]  # local to global path
            #     elif os.path.isdir(p):  # folder
            #         f += glob.iglob(p + os.sep + '*.*')
            #     else:
            #         raise Exception('%s does not exist' % p)
            # self.img_files = sorted([
            #     x.replace('/', os.sep) for x in f
            #     if os.path.splitext(x)[-1].lower() in img_formats
            # ])
        except Exception as e:
            raise Exception('Error loading data from %s: %s\nSee %s' %
                            (path, e, help_url))

        n = len(self.img_files)
        assert n > 0, 'No images found in %s. See %s' % (path, help_url)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.train_transform = A.Compose([
            A.MotionBlur(p=0.25),
            A.RandomRain(p=0.25),
            A.RandomSunFlare(p=0.25),
            A.RandomBrightnessContrast(p=0.25),
            A.HueSaturationValue(hue_shift_limit=hyp['hsv_h'],
                                 sat_shift_limit=hyp['hsv_s'],
                                 val_shift_limit=hyp['hsv_v'],
                                 p=0.25),
        ])

        # Define labels
        # self.label_files = [
        #     x.replace('Inputs', 'Labels').replace('/img', '').replace(
        #         os.path.splitext(x)[-1], '.txt')
        #     for x in self.img_files
        # ]
        # self.dm_files = [
        #     x.replace('img', 'dm').replace(os.path.splitext(x)[-1], '.bin')
        #     for x in self.img_files
        # ]
        # Check cache
        cache_path = str(Path(
            self.label_files[0]).parent) + '.cache'  # cached labels
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files +
                                         self.img_files):  # dataset changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        # Get labels
        labels, shapes = zip(*[cache[x] for x in self.img_files])
        self.shapes = np.array(shapes, dtype=np.float64)
        self.labels = list(labels)

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(
                np.array(shapes) * img_size / stride + pad).astype(
                    np.int) * stride

        # Cache labels
        create_datasubset, extract_bounding_boxes, labels_loaded = False, False, False
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        pbar = tqdm(self.label_files)
        for i, file in enumerate(pbar):
            l = self.labels[i]  # label
            if l.shape[0]:
                assert l.shape[1] == 6, '> 6 label columns: %s' % file
                assert (l >= 0).all(), 'negative labels: %s' % file
                # changed to 2: to adpat depth
                # assert (l[:, 2:] <= 1).all(
                # ), 'non-normalized or out of bounds coordinate labels: %s' % file
                if np.unique(l,
                             axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                if single_cls:
                    l[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = l
                nf += 1  # file found

                # Create subdataset (a smaller dataset)
                if create_datasubset and ns < 1E4:
                    if ns == 0:
                        create_folder(path='./datasubset')
                        os.makedirs('./datasubset/images')
                    exclude_classes = 43
                    if exclude_classes not in l[:, 0]:
                        ns += 1
                        # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
                        with open('./datasubset/images.txt', 'a') as f:
                            f.write(self.img_files[i] + '\n')

                # Extract object detection boxes for a second stage classifier
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    p_di = Path(self.di_files[i])
                    p_m = Path(self.m_files[i])
                    di = np.fromfile(p_di, dtype=np.float16).reshape(
                        img.shape[0], img.shape[1], 2)
                    m = np.fromfile(p_m, dtype=np.bool).reshape(
                        img.shape[0], img.shape[1], 1)
                    img = np.concatenate((img, di[:, :, 0].reshape(
                        img.shape[0], img.shape[1], 1), m),
                                         axis=-1)
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):
                        f = '%s%sclassifier%s%g_%g_%s' % (
                            p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(
                                Path(f).parent)  # make new output folder

                        b = x[1:] * [w, h, w, h]  # box
                        b[2:] = b[2:].max()  # rectangle to square
                        b[2:] = b[2:] * 1.3 + 30  # pad
                        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                        b[[0, 2]] = np.clip(b[[0, 2]], 0,
                                            w)  # clip boxes outside of image
                        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                        assert cv2.imwrite(f, img[
                            b[1]:b[3],
                            b[0]:b[2]]), 'Failure extracting classifier boxes'
            else:
                ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
                # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove

            pbar.desc = 'Scanning labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                cache_path, nf, nm, ne, nd, n)
        if nf == 0:
            s = 'WARNING: No labels found in %s. See %s' % (
                os.path.dirname(file) + os.sep, help_url)
            print(s)
            assert not augment, '%s. Can not train without labels.' % s

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(
                    self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def cache_labels(self, path='labels.cache'):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files),
                    desc='Scanning images',
                    total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                image = Image.open(img)
                image.verify()  # PIL verify
                # _ = io.imread(img)  # skimage verify (from skimage import io)
                shape = exif_size(image)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        l = np.array(
                            [x.split() for x in f.read().splitlines()],
                            dtype=np.float32)  # labels
                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)
                x[img] = [l, shape]
            except Exception as e:
                x[img] = None
                print('WARNING: %s: %s' % (img, e))

        x['hash'] = get_hash(self.label_files + self.img_files)
        # torch.save(x, path)  # save for next time
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        if self.mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            # if random.random() < 0.5:
            #     img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
            #     r = np.random.beta(0.3, 0.3)  # mixup ratio, alpha=beta=0.3
            #     img = (img * r + img2 * (1 - r)).astype(np.uint8)
            #     labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[
                index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img,
                                        shape,
                                        auto=False,
                                        scaleup=self.augment)
            shapes = (h0, w0), (
                (h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:,
                       2] = ratio[0] * w * (x[:, 2] -
                                            x[:, 4] / 2) + pad[0]  # pad width
                labels[:,
                       3] = ratio[1] * h * (x[:, 3] -
                                            x[:, 5] / 2) + pad[1]  # pad height
                labels[:, 4] = ratio[0] * w * (x[:, 2] + x[:, 4] / 2) + pad[0]
                labels[:, 5] = ratio[1] * h * (x[:, 3] + x[:, 5] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                img, labels = random_affine(img,
                                            labels,
                                            degrees=hyp['degrees'],
                                            translate=hyp['translate'],
                                            scale=hyp['scale'],
                                            shear=hyp['shear'])

            # Augment colorspace
            # augment_hsv(img[:, :, :3].astype(np.uint8),
            #             hgain=hyp['hsv_h'],
            #             sgain=hyp['hsv_s'],
            #             vgain=hyp['hsv_v'])
            img[:, :, :3] = self.train_transform(image=img[:, :, :3])["image"]
            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 2:] = xyxy2xywh(labels[:, 2:6])

            # Normalize coordinates 0 - 1
            labels[:, [3, 5]] /= img.shape[0]  # height
            labels[:, [2, 4]] /= img.shape[1]  # width

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 3] = 1 - labels[:, 3]

        labels_out = torch.zeros((nL, 7))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        channels = np.split(img, 5, axis=2)
        img = np.concatenate(
            [channels[2], channels[1], channels[0], channels[3], channels[4]],
            axis=2)  # BGR to RGB
        img = img.transpose(2, 0, 1)  # Put channels first
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def resize_dm(dm, scale):
    h0, w0 = dm.shape[:2]
    indices = np.where(dm[:, :, 1] > 0)
    new_indices = (np.array(indices[0] * scale).astype(np.int),
                   np.array(indices[1] * scale).astype(np.int))
    output = np.zeros((int(round(h0 * scale)), int(round(w0 * scale)), 2))
    output[new_indices[0], new_indices[1], :] = dm[indices[0], indices[1], :]

    return output


def resize_dm_torch(dm, scale):
    """Assume an input tensor of shape (N, 2, H, W)"""
    dm_numpy = dm.detach().cpu().numpy()
    N, C, h0, w0 = dm.shape
    output = torch.empty((N, C, round(h0 * scale), round(w0 * scale)),
                         dtype=dm.dtype,
                         layout=dm.layout,
                         device=dm.device)
    output[:, :, :, :] = 0
    for i in range(N):
        dm_i = dm_numpy[i, :, :, :]
        indices = np.where(dm_i > 0)
        new_indices = (np.array(indices[1] * scale).astype(np.int),
                       np.array(indices[2] * scale).astype(np.int))
        output[i, :, new_indices[0], new_indices[1]] = torch.transpose(
            torch.from_numpy(dm_numpy[i, :, indices[1], indices[2]]), 0,
            1).to(dm.device)

    return output


def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        p = self.img_files[index]
        img = cv2.imread(str(p))
        p_di = self.di_files[index]
        p_m = Path(self.m_files[index])
        di = np.fromfile(p_di,
                         dtype=np.float16).reshape(img.shape[0], img.shape[1],
                                                   2)
        m = np.fromfile(p_m, dtype=np.bool).reshape(img.shape[0], img.shape[1],
                                                    1)
        img = np.concatenate(
            (img, di[:, :, 0].reshape(img.shape[0], img.shape[1], 1), m),
            axis=-1)

        assert img is not None, 'Image Not Found ' + p
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            dm = resize_dm(img[:, :, 3:], r)
            img = cv2.resize(img[:, :, :3].astype(np.uint8),
                             (round(w0 * r), round(h0 * r)),
                             interpolation=interp)
            img = np.concatenate((img, dm), axis=-1)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[
            index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat),
                         cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])


def load_mosaic(self, index):
    # loads images in a mosaic

    labels4 = []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x))
              for x in self.mosaic_border]  # mosaic center x, y
    indices = [index
               ] + [random.randint(0,
                                   len(self.labels) - 1)
                    for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114,
                           dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(
                yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (
                y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc,
                                                         w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b,
                                     x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s,
                out=labels4[:, 1:])  # use with random_affine

        # Replicate
        # img4, labels4 = replicate(img4, labels4)

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    img4, labels4 = random_affine(
        img4,
        labels4,
        degrees=self.hyp['degrees'],
        translate=self.hyp['translate'],
        scale=self.hyp['scale'],
        shear=self.hyp['shear'],
        border=self.mosaic_border)  # border to remove

    return img4, labels4


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(
            0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b,
                                    x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]],
                           axis=0)

    return img, labels


def letterbox_torch(imgs,
                    new_shape=(640, 640),
                    color=(114, 114, 114),
                    auto=True,
                    scaleFill=False,
                    scaleup=True):

    N, c, h_new, w_new = imgs.shape[0], imgs.shape[1], new_shape[0], new_shape[
        1]
    new_imgs = torch.empty((N, c, h_new, w_new),
                           dtype=imgs.dtype,
                           layout=imgs.layout,
                           device=imgs.device)
    imgs = imgs.detach().cpu().numpy()
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = imgs.shape[2:]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(shape[1] * r), int(shape[0] * r)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[
            0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        dm_part = resize_dm_torch(imgs[:, 3:, :, :], r)
        img_part = F.interpolate(imgs[:, :3, :, :],
                                 size=(new_unpad),
                                 mode='bilinear',
                                 align_corners=False)
        imgs = np.concatenate((img_part, dm_part), axis=1)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    rgb_pad_value = 144.0 / 255.0
    new_imgs[:, :3, :, :] = torch.from_numpy(
        np.pad(imgs[:, :3, :, :],
               ((0, 0), (0, 0), (top, bottom), (left, right)),
               constant_values=((0, 0), (0, 0), (rgb_pad_value, rgb_pad_value),
                                (rgb_pad_value, rgb_pad_value))))
    new_imgs[:, 3:, :, :] = torch.from_numpy(
        np.pad(imgs[:, 3:, :, :],
               ((0, 0), (0, 0), (top, bottom), (left, right)),
               constant_values=((0, 0), (0, 0), (0, 0), (0, 0))))
    return new_imgs, ratio, (dw, dh)


def letterbox(img,
              new_shape=(640, 640),
              color=(114, 114, 114),
              auto=True,
              scaleFill=False,
              scaleup=True):

    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(shape[1] * r), int(shape[0] * r)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[
            0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        dm = resize_dm(img[:, :, 3:], r)
        img = cv2.resize(img[:, :, :3].astype(np.uint8),
                         new_unpad,
                         interpolation=cv2.INTER_LINEAR)
        img = np.concatenate((img, dm), axis=-1)

    dm = img[:, :, 3:]
    d = dm[:, :, 0]
    m = dm[:, :, 1]
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    h_new, w_new, c = new_shape[0], new_shape[1], img.shape[2]
    new_img = np.zeros((h_new, w_new, c))
    new_img[:, :, :3] = np.pad(img[:, :, :3],
                               ((top, bottom), (left, right), (0, 0)),
                               constant_values=((144, 144), (144, 144), (0,
                                                                         0)))
    new_img[:, :, 3:] = np.pad(img[:, :, 3:],
                               ((top, bottom), (left, right), (0, 0)),
                               constant_values=((0, 0), (0, 0), (0, 0)))

    return new_img, ratio, (dw, dh)


def multiscale_targets(targets, unpad_shape, pad, new_shape):
    """Given Nx7 targets, where each row is [nb, cls, depth, XYWH]
       where xywh is normalized..
       bring it to the unpadded_shape, and then add padding, then renormalize"""
    device = targets.device
    targets = targets.detach().cpu().numpy()
    xywh = targets[:, 2:]

    h1, w1 = unpad_shape
    h2, w2 = new_shape

    # Rescale
    xywh[:, 0] *= w1
    xywh[:, 2] *= w1
    xywh[:, 1] *= h1
    xywh[:, 3] *= h1

    # Offset by padding
    xywh[:, 0] += pad[0]
    xywh[:, 1] += pad[1]

    # Renormalize
    xywh[:, 0] /= w2
    xywh[:, 2] /= w2
    xywh[:, 1] /= h2
    xywh[:, 3] /= h2

    targets[:, 2:] = xywh
    targets = torch.from_numpy(targets)
    targets = targets.to(device)
    return targets


def random_affine(img,
                  targets=(),
                  degrees=10,
                  translate=.1,
                  scale=.1,
                  shear=10,
                  border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a,
                                    center=(img.shape[1] / 2,
                                            img.shape[0] / 2),
                                    scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[1] + border[
        1]  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[0] + border[
        0]  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi /
                       180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi /
                       180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border[0] != 0) or (border[1] !=
                            0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img,
                             M[:2],
                             dsize=(width, height),
                             flags=cv2.INTER_LINEAR,
                             borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [2, 3, 4, 5, 2, 5, 4, 3]].reshape(
            n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] -
                                                   targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 2) & (h > 2) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 20)

        targets = targets[i]
        targets[:, 2:] = xy[i]

    return img, targets


def cutout(image, labels):
    # https://arxiv.org/abs/1708.04552
    # https://github.com/hysts/pytorch_cutout/blob/master/dataloader.py
    # https://towardsdatascience.com/when-conventional-wisdom-fails-revisiting-data-augmentation-for-self-driving-cars-4831998c5509
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area

        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [
        0.03125
    ] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax,
              xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def reduce_img_size(
        path='../data/sm4/images',
        img_size=1024):  # from utils.datasets import *; reduce_img_size()
    # creates a new ./images_reduced folder with reduced size images of maximum size img_size
    path_new = path + '_reduced'  # reduced images path
    create_folder(path_new)
    for f in tqdm(glob.glob('%s/*.*' % path)):
        try:
            img = cv2.imread(f)
            h, w = img.shape[:2]
            r = img_size / max(h, w)  # size ratio
            if r < 1.0:
                img = cv2.resize(
                    img, (int(w * r), int(h * r)),
                    interpolation=cv2.INTER_AREA)  # _LINEAR fastest
            fnew = f.replace(path,
                             path_new)  # .replace(Path(f).suffix, '.jpg')
            cv2.imwrite(fnew, img)
        except:
            print('WARNING: image failure %s' % f)


def convert_images2bmp():  # from utils.datasets import *; convert_images2bmp()
    # Save images
    formats = [x.lower()
               for x in img_formats] + [x.upper() for x in img_formats]
    # for path in ['../coco/images/val2014', '../coco/images/train2014']:
    for path in ['../data/sm4/images', '../data/sm4/background']:
        create_folder(path + 'bmp')
        for ext in formats:  # ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
            for f in tqdm(glob.glob('%s/*%s' % (path, ext)),
                          desc='Converting %s' % ext):
                cv2.imwrite(
                    f.replace(ext.lower(), '.bmp').replace(path, path + 'bmp'),
                    cv2.imread(f))

    # Save labels
    # for path in ['../coco/trainvalno5k.txt', '../coco/5k.txt']:
    for file in ['../data/sm4/out_train.txt', '../data/sm4/out_test.txt']:
        with open(file, 'r') as f:
            lines = f.read()
            # lines = f.read().replace('2014/', '2014bmp/')  # coco
            lines = lines.replace('/images', '/imagesbmp')
            lines = lines.replace('/background', '/backgroundbmp')
        for ext in formats:
            lines = lines.replace(ext, '.bmp')
        with open(file.replace('.txt', 'bmp.txt'), 'w') as f:
            f.write(lines)


def recursive_dataset2bmp(
    dataset='../data/sm4_bmp'
):  # from utils.datasets import *; recursive_dataset2bmp()
    # Converts dataset to bmp (for faster training)
    formats = [x.lower()
               for x in img_formats] + [x.upper() for x in img_formats]
    for a, b, files in os.walk(dataset):
        for file in tqdm(files, desc=a):
            p = a + '/' + file
            s = Path(file).suffix
            if s == '.txt':  # replace text
                with open(p, 'r') as f:
                    lines = f.read()
                for f in formats:
                    lines = lines.replace(f, '.bmp')
                with open(p, 'w') as f:
                    f.write(lines)
            elif s in formats:  # replace image
                cv2.imwrite(p.replace(s, '.bmp'), cv2.imread(p))
                if s != '.bmp':
                    os.system("rm '%s'" % p)


def imagelist2folder(path='data/coco_64img.txt'
                     ):  # from utils.datasets import *; imagelist2folder()
    # Copies all the images in a text file (list of images) into a folder
    create_folder(path[:-4])
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            os.system('cp "%s" %s' % (line, path[:-4]))
            print(line)


def create_folder(path='./new_folder'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
