import rosbag
import os
from tqdm import tqdm
import cv2
from cv_bridge import CvBridge
import numpy as np
import shutil
from pyproj import Proj

from utils.utils import convert_msg_to_numpy
from utils.utils import get_driving_interval


class RosbagExtractor:
    def __init__(self, cfg):
        self.point_cloud_file_format = cfg.point_cloud_file_format
        self.moving_only_flag = cfg.moving_only
        self.rosbag_file_path = cfg.rosbag_file_path
        self.rosbag_filename = os.path.basename(cfg.rosbag_file_path)
        self.timestamp_started_driving = None
        self.timestamp_stopped_driving = None

        # Folder where the extracted data should go to
        self.data_folder = os.path.join(
            os.path.dirname(cfg.rosbag_file_path), "../data",
            os.path.splitext(self.rosbag_filename)[0])

        # Generate Bag object and extract meta data of all topics
        self.bag = rosbag.Bag(cfg.rosbag_file_path)
        self.type_and_topic_info = self.bag.get_type_and_topic_info(
            topic_filters=None)

        # If the moving_only_flag is true, extract from /tf topic first by
        # putting it at the first position
        if cfg.moving_only:
            if "/tf" in cfg.topics:
                try:
                    cfg.topics.remove("/tf")
                except:
                    pass

                tmp_list = ["/tf"]
                tmp_list.extend(cfg.topics)
                cfg.topics = tmp_list

        # Create a dictionary to correspond topic names to folder names
        self.topic_name_to_folder_name_dict = {
            "/sensors/fw_lidar/point_cloud_raw": "fw_lidar",
            "/sensors/mrh_lidar/point_cloud_raw": "mrh_lidar",
            "/sensors/right_camera/image_color": "right_camera",
            "/sensors/forward_camera/image_color": "forward_camera",
            "/sensors/left_camera/image_color": "left_camera",
            "/tf": "tf",
            "/pilatus_can/GNSS": "gnss"
        }

        # Create a dictionary to correspond topic names to data types
        self.topic_name_to_data_type_dict = {
            "/sensors/fw_lidar/point_cloud_raw": "fw_lidar_pcs",
            "/sensors/mrh_lidar/point_cloud_raw": "mrh_lidar_pcs",
            "/sensors/right_camera/image_color": "right_camera_imgs",
            "/sensors/forward_camera/image_color": "forward_camera_imgs",
            "/sensors/left_camera/image_color": "left_camera_imgs",
            "/tf": "transformations",
            "/pilatus_can/GNSS": "gnss_data"
        }

    def init_file_structure(self):

        if os.path.exists(self.data_folder):
            print(
                "The directory {} exists already indicating that the rosbag has already been extracted before."
                .format(os.path.splitext(self.rosbag_filename)[0]))

            print("Reextracting rosbag. Cleaning old data.")
            shutil.rmtree(self.data_folder)

        else:
            os.makedirs(self.data_folder)

        return 1

    def extract(self, topic):
        print("Started extraction of topic {} in {}.".format(
            topic, self.rosbag_filename))

        msg_type = self.type_and_topic_info[1][str(topic)].msg_type

        if msg_type == "sensor_msgs/PointCloud2":
            print("Extracting point clouds, {} format".format(
                self.point_cloud_file_format))
            pcs, timestamps = self.extract_sensor_msgs_point_cloud_2(topic)

        elif msg_type == "sensor_msgs/Image":
            print("Extracting images")
            images, timestamps = self.extract_sensor_msgs_image(topic)

        elif msg_type == "tf2_msgs/TFMessage":
            print("Extracting tf2 transformations")
            transforms_dict = self.extract_tf2_msgs_tf_message(topic)

        elif msg_type == "pilatus_can/GNSS":
            print("Extracting pilatus_can/GNSS")
            poses, timestamps = self.extract_pilatus_can_gnss(topic)

    def extract_sensor_msgs_point_cloud_2(self, topic):
        pbar = tqdm(total=self.type_and_topic_info[1][topic].message_count,
                    desc=topic)

        data_dir = os.path.join(self.data_folder,
                                self.topic_name_to_folder_name_dict[topic])
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        counter = 0
        timestamps = []
        pcs = []
        for _, msg, time in self.bag.read_messages(topics=[topic]):
            # Weird issue is that time and msg.header.stamp differ (at around 0.1s)
            # Taking msg.header.stamp for now

            pbar.update(1)

            timestamp = float("{}.{}".format(
                str(msg.header.stamp.secs),
                str(msg.header.stamp.nsecs).zfill(9)))
            if self.moving_only_flag:
                if timestamp < self.timestamp_started_driving or timestamp > self.timestamp_stopped_driving:
                    continue

            pc = convert_msg_to_numpy(msg)
            if pc.size == 0:
                continue

            if self.point_cloud_file_format == 'npy':
                np.save(os.path.join(data_dir, str(counter).zfill(8)), pc)
            elif self.point_cloud_file_format == 'bin':
                pc.tofile(
                    os.path.join(data_dir,
                                 str(counter).zfill(8) + '.bin'))
            else:
                print(
                    "Saving in specified point cloud format is not possible.")

            pcs.append(pc)
            timestamps.append(timestamp)

            counter += 1

        pbar.close()

        with open(os.path.join(data_dir, 'timestamps.txt'), 'w') as filehandle:
            filehandle.writelines("{:.6f}\n".format(timestamp)
                                  for timestamp in timestamps)

        return pcs, timestamps

    def extract_sensor_msgs_image(self, topic):
        pbar = tqdm(total=self.type_and_topic_info[1][topic].message_count,
                    desc=topic)

        data_dir = os.path.join(self.data_folder,
                                self.topic_name_to_folder_name_dict[topic])
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        counter = 0
        timestamps = []
        images = []
        bridge = CvBridge()

        for _, msg, _ in self.bag.read_messages(topics=[topic]):
            pbar.update(1)

            timestamp = float("{}.{}".format(
                str(msg.header.stamp.secs),
                str(msg.header.stamp.nsecs).zfill(9)))
            if self.moving_only_flag:
                if timestamp < self.timestamp_started_driving or timestamp > self.timestamp_stopped_driving:
                    continue

            image = bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv2.imwrite(os.path.join(data_dir,
                                     str(counter).zfill(8) + '.png'), image)
            timestamps.append(timestamp)
            images.append(image)

            counter += 1

        pbar.close()

        with open(os.path.join(data_dir, 'timestamps.txt'), 'w') as filehandle:
            filehandle.writelines("{:.6f}\n".format(timestamp)
                                  for timestamp in timestamps)

        return images, timestamps

    def extract_tf2_msgs_tf_message(self, topic):
        pbar = tqdm(total=self.type_and_topic_info[1][topic].message_count,
                    desc=topic)

        data_dir = os.path.join(self.data_folder,
                                self.topic_name_to_folder_name_dict[topic])
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        transforms_dict = {}

        for _, msg, _ in self.bag.read_messages(topics=[topic]):
            pbar.update(1)

            for transform in msg.transforms:

                # Seperate different transformations from each other
                key = transform.child_frame_id + "_to_" + transform.header.frame_id
                if not transforms_dict.has_key(key):
                    transforms_dict[key] = []

                t = []
                t.append(
                    float("{}.{}".format(
                        str(transform.header.stamp.secs),
                        str(transform.header.stamp.nsecs).zfill(9))))
                t.append(transform.transform.translation.x)
                t.append(transform.transform.translation.y)
                t.append(transform.transform.translation.z)
                t.append(transform.transform.rotation.x)
                t.append(transform.transform.rotation.y)
                t.append(transform.transform.rotation.z)
                t.append(transform.transform.rotation.w)

                transforms_dict[key].append(t)

        pbar.close()

        # Extract timestamps when car started and stopped driving for filtering
        if self.moving_only_flag:
            self.timestamp_started_driving, self.timestamp_stopped_driving = get_driving_interval(
                transforms_dict["egomotion_to_world"])

            # Filter transform_dict using timestamp filter
            for key in transforms_dict:
                driving_interval_mask = [
                    transform[0] > self.timestamp_started_driving
                    and transform[0] < self.timestamp_stopped_driving
                    for transform in transforms_dict[key]
                ]
                transforms_dict[key] = (np.array(
                    transforms_dict[key])[driving_interval_mask]).tolist()

        for key in transforms_dict:
            with open(os.path.join(data_dir, key + '.txt'), 'w') as filehandle:
                filehandle.writelines(
                    "{:.6f}, {}, {}, {}, {}, {}, {}, {}\n".format(
                        transform[0],
                        transform[1],
                        transform[2],
                        transform[3],
                        transform[4],
                        transform[5],
                        transform[6],
                        transform[7],
                    ) for transform in transforms_dict[key])

        return transforms_dict

    def extract_pilatus_can_gnss(self, topic):
        pbar = tqdm(total=self.type_and_topic_info[1][topic].message_count,
                    desc=topic)

        data_dir = os.path.join(self.data_folder,
                                self.topic_name_to_folder_name_dict[topic])
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        proj = Proj(proj='utm', zone=32, ellps='WGS84', preserve_units=False)
        counter = 0
        timestamps = []
        poses = []

        for _, msg, _ in self.bag.read_messages(topics=[topic]):
            pbar.update(1)

            timestamp = float("{}.{}".format(
                str(msg.header.stamp.secs),
                str(msg.header.stamp.nsecs).zfill(9)))
            if self.moving_only_flag:
                if timestamp < self.timestamp_started_driving or timestamp > self.timestamp_stopped_driving:
                    continue

            lat = msg.RTK_latitude
            long = msg.RTK_longitude
            roll = msg.INS_roll
            pitch = msg.INS_pitch
            heading = msg.dual_heading

            projection = proj(long, lat)
            curr_pose = [projection[0], projection[1], pitch, roll, heading]
            poses.append(curr_pose)
            timestamps.append(timestamp)
            counter += 1

        pbar.close()

        print(self.moving_only_flag)
        vehicle_pos = np.asarray(poses)
        print(vehicle_pos.shape)
        print(np.max(vehicle_pos[:, 0]))
        print(np.min(vehicle_pos[:, 1]))
        np.save('vehicle_pos.npy', vehicle_pos)

        with open(os.path.join(data_dir, 'timestamps.txt'), 'w') as filehandle:
            filehandle.writelines("{:.6f}\n".format(timestamp)
                                  for timestamp in timestamps)

        return poses, timestamps
