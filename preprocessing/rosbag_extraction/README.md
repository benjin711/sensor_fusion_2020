## Rosbag Extraction Tools

This folder contains scripts to extract data from rosbags. I suggest to seperate extraction and processing to seperate folders since then, we can use another virtual environment with python3 for processing. As python2 is used, it is recommended to use the virtualenv tool to create the virtual environment here. The --system-site-packages flag must be provided when installing the virtual environment because we rely on system site packages like the "rosbag" package, which is seemingly only available with a ROS installation and can't be installed using e.g. pip. To install the virtual environment with system site packages available, do: 

```
python2 -m virtualenv --system-site-packages venv
```

### Requirements
The rosbags and gtmd.csv files are expected to be in a folder structure similar to the following for extraction. The extracted data will be in a folder called "data" in the folders of the respective testing days alongside the "gtmd" and "rosbags" folders. 
```
amz_sensor_fusion_data/
├── 2020-07-05_tuggen
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
```
There is no requirements.txt file because with the --system-site-packages flag, all potentially unnecessary system site packages get displayed as well. Here is a list of packages that work for me:

opencv-python 4.2.0.32
tqdm 4.39.0
cv-bridge 1.13.0
rosbag 1.14.6
numpy 1.13.3

### Usage 
```
python preprocessing/rosbag_extraction/main.py -h

python preprocessing/rosbag_extraction/main.py -r <rosbag_file_path> 
```

### Issues & Next Steps
- It should be possible to specify several rosbags to extract from, instead of just one at a time

