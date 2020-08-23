## Data Processing Tools

### Done
- Cluster images from the three cameras into triples with the same time stamp. Add black images to triples when one of the cameras failed to capture an image. Use the timestamp of the image triple as reference timestamp for the point clouds. 
- Match the time-wise closest point clouds from mrh and fw lidar to the the image triple. Add an empty point cloud if there are no point clouds close to an image triples time stamp. Motion compensate the point clouds and transform them into the mrh_frame. The result of this are 3x images (in forward_camera, left_camera, right_camera frames) + 2x point cloud (both in the mrh_lidar frame) data quintuples all with their respective time stamp.

### Todos/Issues
- The egomotion compensation of the point clouds takes a very long time ~6h per rosbag (makes it also tedious to debug)
- The fw_lidar to mrh_lidar calibration is off for most rosbags, we need to manually recalibrate the point clouds
- Extraction of the /pilatus_can/GNSS topic is not yet implemented (we should probably use that instead of relying on velocity estimation + slam for getting the car position). The topic should give us an additional heading information compared to the GTMD data.
- Convert the point cloud from the mrh_lidar to the cylindrical coordinate frame
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

To filter and group the data according to the timestamps, do:
```
pipenv shell 
python main.py -h
python main.py -d <path to folder where the extracted data is stored in e.g. autocross_2020-07-05-12-35-31> --match_data
python main.py -d <path to folder where the extracted data is stored in e.g. autocross_2020-07-05-12-35-31> --match_data --keep_orig_data_folders
        
```
The --keep_orig_data_folders flag helps debugging, since intermediate folders/files are not deleted.