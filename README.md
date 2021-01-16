# Sensor Fusion 2020

This repository organizes the code to develop the early sensor fusion pipeline for AMZ driverless 2020. This project is a joint semester project authored by Carter Fang, Jiapeng Zhong, Benjamin Jin and supervised by Dr. Martin Oswald from the computer vision and geometry group.

---
## Overview

The goal of the project is to develop a robust and accurate cone detection algorithm by fusing raw perception data provided by the perception sensor modalities of *pilatus*, a Hesai 20B LiDAR on the main roll hoop (MRH), a Hesai 64 on the front wing (FW) and three Basler acA2500-gc20 cameras arranged arount the MRH. The motion compensated point clouds and images fused into a cylindrical coordinate system and are then jointly fed into a Yolo-like network architecture which infers bounding boxes around the cones and the distance of the cones. The bounding box and distance is then used to infer the cone positions relative to the car. 

## Model
The model implementation is based on 
[YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4), 
described in [this paper](https://arxiv.org/abs/2004.10934).

Forked on 2020/10/28.

## Usage test.py
Python Version 3.8 has been used on leonhard. On leonhard, use the command ```module load gcc/6.3.0 python_gpu/3.8.5```. Locally, I have been using pipenv virtual environments with python version 3.6, but pipenv doesn't seem to work well on leonhard. On leonhard I just installed all necessary dependencies globally. Pipfiles can be found in the repo. Files are formatted using yapf (at least the ones that I modified). 

Models are in the perception folder on gdrive as well as the NAS in pilatus-2021/sensor_fusion_model_weights and need to be downloaded e.g. into the weights folder. The perception folder on the gdrive also contains a thorough evaluation of the model. Inference can be run by using the test.py script on leonhard or locally. Set the path to the model weights accordingly, then run e.g.:
```
bsub -W 4:00 -n 20 -R "rusage[mem=4500, ngpus_excl_p=1]" python test.py --data data/amz_data_splits.yaml --weights weights/best_exp06.pt --batch-size 1 --img-size 1280 --iou-thres 0.3 --conf-thres 0.01 --task test --merge --generate-depth-stats --device 0 --save-pkl
python test.py --data data/amz_data_splits.yaml --weights weights/best_exp06.pt --save-pkl

```
## Usage train.py
Training can be started locally and on leonhard.
```
bsub -W 4:00 -n 20 -R "rusage[mem=4500,ngpus_excl_p=8]" python train.py --batch-size 32 --data data/amz_data_splits.yaml --noautoanchor --cfg dels/yolov4m-rgbd.yaml --rect
python train.py --batch-size 2 --data data/amz_data_splits.yaml --noautoanchor --cfg models/yolov4m-rgbd.yaml --rect --epochs 1
```

## Usage evaluation scripts
For information on the evaluation scripts refer to the the README.md in the evaluation folder.