import rosbag
import os
from tqdm import tqdm
import cv2
from cv_bridge import CvBridge
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

        # Create a dictionary to store the extracted data
        # self.extracted_rosbag_data_dict = {}

    def extract(self, topic):
        print("Started extraction of topic {} in {}.".format(
            topic, self.rosbag_filename))

        msg_type = self.type_and_topic_info[1][str(topic)].msg_type

        if msg_type == "sensor_msgs/PointCloud2":
            print("Extracting point clouds")
            self.extract_sensor_msgs_point_cloud_2(topic)
        elif msg_type == "sensor_msgs/Image":
            print("Extracting images")
            self.extract_sensor_msgs_image(topic)
        elif msg_type == "tf2_msgs/TFMessage":
            print("Extracting tf2 transformations")
            self.extract_tf2_msgs_tf_message(topic)
        elif msg_type == "pilatus_can/GNSS":
            print("Extracting pilatus_can/GNSS")
            self.extract_pilatus_can_gnss(topic)

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

            pc = convert_msg_to_numpy(msg)
            if pc.size == 0:
                continue

            np.save(os.path.join(data_dir, str(counter).zfill(8)), pc)
            pcs.append(pc)
            timestamps.append(msg.header.stamp)

            counter += 1

        pbar.close()

        with open(os.path.join(data_dir, 'timestamps.txt'), 'w') as filehandle:
            filehandle.writelines(
                "{}.{}\n".format(str(timestamp.secs),
                                 str(timestamp.nsecs).zfill(9))
                for timestamp in timestamps)
        
        return None

    def extract_sensor_msgs_image(self, topic):
        pbar = tqdm(total=self.type_and_topic_info[1][topic].message_count,
                    desc=topic)

        data_dir = os.path.join(self.data_folder,
                                self.topic_name_to_folder_name_dict[topic])
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        counter = 0
        timestamps = []
        bridge = CvBridge()

        for _, msg, _ in self.bag.read_messages(topics=[topic]):
            pbar.update(1)

            image = bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv2.imwrite(os.path.join(data_dir,
                                     str(counter).zfill(8) + '.png'), image)
            timestamps.append(msg.header.stamp)

            counter += 1

        pbar.close()

        with open(os.path.join(data_dir, 'timestamps.txt'), 'w') as filehandle:
            filehandle.writelines(
                "{}.{}\n".format(str(timestamp.secs),
                                 str(timestamp.nsecs).zfill(9))
                for timestamp in timestamps)

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
                t.append(transform.child_frame_id)
                t.append(transform.header.frame_id)
                t.append("{}.{}".format(
                    str(transform.header.stamp.secs),
                    str(transform.header.stamp.nsecs).zfill(9)))
                t.append(transform.transform.translation.x)
                t.append(transform.transform.translation.y)
                t.append(transform.transform.translation.z)
                t.append(transform.transform.rotation.x)
                t.append(transform.transform.rotation.y)
                t.append(transform.transform.rotation.z)
                t.append(transform.transform.rotation.w)

                transforms_dict[key].append(t)

        pbar.close()

        for key in transforms_dict:
            with open(os.path.join(data_dir, key + '.txt'), 'w') as filehandle:
                filehandle.writelines(
                    "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
                        transform[0],
                        transform[1],
                        transform[2],
                        transform[3],
                        transform[4],
                        transform[5],
                        transform[6],
                        transform[7],
                        transform[8],
                        transform[9],
                    ) for transform in transforms_dict[key])

    def extract_pilatus_can_gnss(self, topic):
        print("Not implemented.")
        pass

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['bag']
        return state

    def __setstate__(self, state):
        # Restore instance attributes 
        self.__dict__.update(state)
        # Restore the previously opened file's state. To do so, we need to
        # reopen the bag file
        self.bag = rosbag.Bag(self.rosbag_file_path)
