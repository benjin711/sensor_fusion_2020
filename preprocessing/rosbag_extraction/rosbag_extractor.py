import rosbag
from tqdm import tqdm


class RosbagExtractor:
    def __init__(self, rosbag_filename):
        self.rosbag_filename = rosbag_filename
        self.bag = rosbag.Bag(rosbag_filename)
        self.type_and_topic_info = self.bag.get_type_and_topic_info(
            topic_filters=None)

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
        pbar = tqdm(total=self.bag.get_message_count(
            topic_filters=[topic]), desc=topic)

        for _, msg, time in self.bag.read_messages(topics=[topic]):
            pbar.update(1)

    def extract_sensor_msgs_image(self, topic):
        pass

    def extract_tf2_msgs_tf_message(self, topic):
        pass

    def extract_pilatus_can_gnss(self, topic):
        pass
