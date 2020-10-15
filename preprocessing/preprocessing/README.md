## Data Processing Tools

### Done
- Cluster images from the three cameras into triples with the same time stamp. Add black images to triples when one of the cameras failed to capture an image. Use the timestamp of the image triple as reference timestamp for the point clouds, rtk car position + heading and cones. 
- Matching of cones and RTK car positions to the reference time stamps
- Match the time-wise closest point clouds from mrh and fw lidar to the the image triple. Add an empty point cloud if there are no point clouds close to an image triples time stamp. Motion compensate the point clouds and transform them into the mrh_frame. The result of this are 3x images (in forward_camera, left_camera, right_camera frames) + 2x point cloud (both in the mrh_lidar frame) data quintuples all with their respective time stamp.

### Todos/Issues
- Convert the point cloud from the mrh_lidar to the cylindrical coordinate frame
- Convert the images to the yet to be defined frame and convert them to cylindrical coordinates 

### Usage
Virtual environment can be created using pipenv.
```
pipenv install
```

Requirement is that the images, point clouds, tf transforms and car GNSS have already been extracted using the rosbag extraction scripts. A file structure like the following is expected:
```
.
├── 2020-07-05_tuggen
│   ├── data
│   │   ├── autocross_2020-07-05-11-58-07
│   │   ├── autocross_2020-07-05-12-35-31
│   │   └── autocross_2020-07-05-13-57-26
│   ├── gtmd
│   │   └── 2020-07-05_tuggen.csv
│   ├── rosbags
│   │   ├── autocross_2020-07-05-11-58-07.bag
│   │   ├── autocross_2020-07-05-12-35-31.bag
│   │   └── autocross_2020-07-05-13-57-26.bag
│   ├── static_transformations
│   │   └── static_transformations.yaml
│   └── sensor_extrinsics
│       ├── extrinsics_lidar_forward.yaml
│       ├── extrinsics_lidar_left.yaml
│       └── extrinsics_lidar_right.yaml
├── 2020-07-08_duebendorf
│   ├── data
│   │   └── autocross_2020-07-08-09-53-46
│   ├── gtmd
│   │   ├── 2020-07-08_duebendorf_edited.csv
│   │   └── 2020-07-08_duebendorf_faulty.csv
│   ├── rosbags
│   │   └── autocross_2020-07-08-09-53-46.bag
│   └── static_transformations
│       └── static_transformations.yaml
...
```


To preprocess a single data folder from one rosbag, do:
```
pipenv shell 
python main.py -h
python main.py -d <path to folder where the extracted data is stored in e.g. autocross_2020-07-05-12-35-31> --match_data
python main.py -d <path to folder where the extracted data is stored in e.g. autocross_2020-07-05-12-35-31> --match_data --keep_orig_data_folders   
```
The --keep_orig_data_folders flag helps debugging, since intermediate folders/files are not deleted.

To preprocess all data folders within the base folder, do:
```
pipenv shell 
python main.py --preprocess_all -b <path to the base folder with the test day folders inside> --match_data
python main.py --preprocess_all -b <path to the base folder with the test day folders inside> --match_data --keep_orig_data_folders  
```