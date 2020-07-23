import rosbag
import os
from tqdm import tqdm
import numpy as np
from utils.pc_utils import convert_msg_to_numpy


class RosbagExtractor:
    def __init__(self, rosbag_file_path):
        self.rosbag_file_path = rosbag_file_path
        self.rosbag_filename = os.path.basename(rosbag_file_path)
        self.bag = rosbag.Bag(rosbag_file_path)
        self.type_and_topic_info = self.bag.get_type_and_topic_info(
            topic_filters=None)

        # Create a folder where the extracted data should go to
        self.data_folder = os.path.join(
            os.path.dirname(rosbag_file_path), "../data",
            os.path.splitext(self.rosbag_filename)[0])
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

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

    def extract(self, topic):
        print("Started extraction of topic {} in {}.".format(
            topic, self.rosbag_filename))

        msg_type = self.type_and_topic_info[1][str(topic)].msg_type

        if msg_type == "sensor_msgs/PointCloud2":
            print("Extracting point clouds")
            self.extract_sensor_msgs_point_cloud_2(topic)
        elif msg_type == "sensor_msgs/Image":
            print("Extracting images")
        elif msg_type == "tf2_msgs/TFMessage":
            print("Extracting tf2 transformations")
        elif msg_type == "pilatus_can/GNSS":
            print("Extracting pilatus_can/GNSS")

    def extract_sensor_msgs_point_cloud_2(self, topic):
        pbar = tqdm(total=self.type_and_topic_info[1][topic].message_count,
                    desc=topic)

        data_dir = os.path.join(self.data_folder,
                                self.topic_name_to_folder_name_dict[topic])
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        counter = 0
        timestamps = []
        for _, msg, time in self.bag.read_messages(topics=[topic]):
            # Weird issue is that time and msg.header.stamp differ (at around 0.1s)
            # Taking msg.header.stamp for now

            pbar.update(1)

            pc = convert_msg_to_numpy(msg)
            if pc.size == 0:
                continue

            np.save(os.path.join(data_dir, str(counter).zfill(8)), pc)
            timestamps.append(msg.header.stamp)

            counter += 1

        pbar.close()
        bag.close()

        with open(os.path.join(data_dir, 'timestamps.txt'), 'w') as filehandle:
            filehandle.writelines("{}\n".format(timestamp)
                                  for timestamp in timestamps)

    def extract_sensor_msgs_image(self, topic):
        pass

    def extract_tf2_msgs_tf_message(self, topic):
        pass

    def extract_pilatus_can_gnss(self, topic):
        pass
