## Data Processing Tools

### Done
- Cluster images from the three cameras into triples with the (approximately) same timestamp. Approach: take the images of the forward camera as references. For each image of the forward camera, see if there are images of the left and right  camera within +-0.001s. If yes, then take the three images as a triple, if the time interval in which the images were taken is less than 0.001s. Take the mean time stamp as the new reference time stamp. 

### Todos
- Find the closest point clouds from mrh and fw lidar timestamp-wise to every image triple
- Do egomotion compensation of the point clouds into a yet to defined frame at the timestamp of the image triple
- Convert the point cloud to cylindrical coordinates
- Convert the images to the yet to be defined frame and convert them to cylindrical coordinates 

### Usage
Virtual environment can be created using pipenv.
```
pipenv install
```

Requirement is that the images, point clouds and transforms have already been extracted using the rosbag extraction scripts. A file structure like the following is expected:
.
├── 2020-07-05_tuggen
│   ├── data
│   │   └── autocross_2020-07-05-12-35-31
│   │       ├── forward_camera
│   │       ├── fw_lidar
│   │       ├── left_camera
│   │       ├── mrh_lidar
│   │       ├── right_camera
│   │       └── tf
│   ├── gtmd
│   │   └── 2020-07-05_tuggen.csv
│   └── rosbags
│       ├── autocross_2020-07-05-12-35-31.bag
│       └── autocross_2020-07-05-18-13-26.bag
└── 2020-07-08_duebendorf
    ├── gtmd
    │   └── 2020-07-08_duebendorf.csv
    └── rosbags
        └── autocross_2020-07-08-09-02-59.bag

To filter the images and group them according to the timestamps, do:
```
pipenv shell 
python main.py -h
python main.py -d <path to folder where the extracted data is stored in e.g. autocross_2020-07-05-12-35-31> --match_images
        
```