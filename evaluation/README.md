# Evaluation of the predicted 3D cone positions

The eval_<...> scripts serve to evaluate predicted 3D cone positions of different pipelines. The ground truth cones are obtained from RTK systems on pilatus and the GTMD and elaborate postprocessing is done to transform these cone positions into the relevant car coordinate frames. For each predicted 3D cone, we try to find the corresponding ground truth cone using a nearest neighbor search. Then the average distance error, standard deviation of the error and the number of predictions is calculated for distance ranges e.g. 0 - 4m, 4 - 8m, ... and displayed in a bar diagram.

## As predicted from the sensor fusion network
The test.py file provides the inference data in an inference.pkl file. The eval_sf_cone_positions.py takes the predictions and the ground thruth cones from the sensor_fusion_data folder and visualizes the predictions and ground truth in BEV perspective and image perspective. The script also calculates the average distance error of the predictions per distance range (0 - 3m, 3 - 6m etc.). To learn more about necessary flags:
```
python eval_sf_cone_positions.py -h
```

## As predicted from the lidar pipeline/local map 
The rosbag_extractor.py and data_preprocessor.py scripts can extract the lidar/local map cone array predictions and match them to the reference timestamps of the images. Hence it's possible to compare these cone arrays, which can be found in the lidar_cone_arrays_filtered/lm_cone_arrays_filtered folder, with the ground truth cone arrays. The eval_lidar_cone_positions.py script visualizes lidar/local map pipeline predictions together with the ground truth cone arrays in the BEV perspective and also calculates the average distance error bar diagram. 
```
python eval_cone_positions.py -h
```