## Data Processing Tools

### Done
- Cluster images from the three cameras into triples with the (approximately) same timestamp
  - Approach: take the images of the forward camera as references. For each image of the forward camera, see if there are images of the left and right  camera within +-0.001s. If yes, then take the three images as a triple, if the time interval in which the images were taken is less than 0.001s. Take the mean time stamp as the new reference time stamp. 

### Todos
- Find the closest point clouds from mrh and fw lidar timestamp-wise to every image triple
- Do egomotion compensation of the point clouds into a yet to defined frame at the timestamp of the image triple
- Convert the point cloud to cylindrical coordinates
- Convert the images to the yet to be defined frame and convert them to cylindrical coordinates
- Store the 