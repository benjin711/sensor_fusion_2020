### Rosbag Extraction Tools

This folder contains scripts to extract and process data from rosbags. A virtual environment managed by pipenv is provided. The --site-packages flag must be provided when installing the virtual environment because we rely on system site packages like "rosbag" which is seemingly only available with a ROS installation and can't be installed using e.g. pip. To install the virtual environment with system site packages available: 

```
pipenv install --site-packages
```