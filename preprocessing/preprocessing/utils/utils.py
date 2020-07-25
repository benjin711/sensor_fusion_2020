def get_camera_timestamps():
    timestamp_filepaths_dict = {
        "forward_camera":
        os.path.join(self.data_folder_path, "forward_camera/timestamps.txt"),
        "left_camera":
        os.path.join(self.data_folder_path, "left_camera/timestamps.txt"),
        "right_camera":
        os.path.join(self.data_folder_path, "right_camera/timestamps.txt")
    }

    return read_timestamps(timestamp_filepaths_dict)


def read_timestamps(timestamp_filepaths_dict):
    timestamp_arrays_dict = {}
    for key in timestamp_filepaths_dict:
        timestamp_arrays_dict[key] = []

    for key in timestamp_filepaths_dict:
        with open(timestamp_filepaths_dict[key]) as timestamps_file:
            for timestamp in timestamps_file:
                timestamp_arrays_dict[key].append(timestamp)
            timestamp_arrays_dict[key] = np.array(timestamp_arrays_dict[key],
                                                  dtype=np.float)
    return timestamp_arrays_dict