## Sensor Fusion 2020

This repository organizes the code to develop the early sensor fusion pipeline for AMZ driverless 2020. This project is a joint semester project authored by Carter Fang, Jiapeng Zhong, Benjamin Jin and supervised by Dr. Martin Oswald from the computer vision and geometry group.

---
** Overview **

The goal of the project is to develop a robust and accurate cone detection algorithm by fusing raw perception data provided by the perception sensor modalities of *pilatus*, a Hesai 20B LiDAR on the main roll hoop (MRH), a Hesai 64 on the front wing (FW) and three Basler acA2500-gc20 cameras arranged arount the MRH. The motion compensated point clouds and images fused into a cylindrical coordinate system and are then jointly fed into a Yolo-like network architecture which infers bounding boxes around the cones and the distance of the cones. The bounding box and distance is then used to infer the cone positions relative to the car. 