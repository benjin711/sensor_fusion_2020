## Rosbag Extraction Tools

This folder contains scripts to extract data from rosbags. As python2 is required, it is recommended to use the virtualenv tool to create the virtual environment here. The --system-site-packages flag must be provided when installing the virtual environment because we rely on system site packages like the "rosbag" package, which is seemingly only available with a ROS installation and can't be installed using e.g. pip. To install the virtual environment with system site packages available, do: 

```
python2 -m virtualenv --system-site-packages venv
```

### Requirements
The rosbags, gtmd.csv and static transformation files are expected to be in a folder structure like following for extraction. The extracted data will be in a folder called "data" in the folders of the respective testing days alongside the "gtmd", "rosbags" and "static_transformations" folders. 
```
sensor_fusion_data/
├── 2020-07-05_tuggen
│   ├── gtmd
│   │   └── 2020-07-05_tuggen.csv
│   ├── rosbags
│   │   ├── autocross_2020-07-05-11-58-07.bag
│   │   ├── autocross_2020-07-05-12-35-31.bag
│   │   └── autocross_2020-07-05-13-57-26.bag
│   └── static_transformations
│       └── static_transformations.yaml
├── 2020-07-08_duebendorf
│   ├── gtmd
│   │   ├── 2020-07-08_duebendorf_edited.csv
│   │   └── 2020-07-08_duebendorf_faulty.csv
│   ├── rosbags
│   │   └── autocross_2020-07-08-09-53-46.bag
│   └── static_transformations
│       └── static_transformations.yaml

```

There is no requirements.txt file because with the --system-site-packages flag, all potentially unnecessary system site packages get displayed as well. Here is a list of packages that work for me:

opencv-python 4.2.0.32
tqdm 4.39.0
cv-bridge 1.13.0
rosbag 1.14.6
numpy 1.13.3

### Usage 
To extract from one specific rosbag, do:
```
source venv/bin/activate
python -r <rosbag_file_path> 
```

To extract from all rosbags within the base folder, do:
```
source venv/bin/activate
python --extract_all -b <base_folder>
```

Check the config.py file to see a list of topics that are currently supported to be extracted.

