from pyproj import Proj
import csv
from matplotlib import pyplot as plt


def parse_gtmd_csv(input_path, output_path='./gtmd_output.csv', plot=False):
    """
    Parse the gtmd csv where each row is formatted:
    [color, latitude, longitude, height, variance of position]
    output a csv where each row is formatted:
    [color, x position, y position, variance]

    :param input_path: path (string) to input csv
    :param output_path: path (string) to output csv
    """

    proj = Proj(proj='utm', zone=32, ellps='WGS84', preserve_units=False)
    x_arr, y_arr = [], []

    with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
        csv_reader = csv.reader(input_file, delimiter=',')
        next(csv_reader) # skip header

        csv_writer = csv.writer(output_file, delimiter=',')
        output_header = ['Colour', 'X', 'Y', 'Variance']
        csv_writer.writerow(output_header)

        for idx, row in enumerate(csv_reader, 1):
            colour = row[0]
            lat = float(row[1])
            long = float(row[2])
            variance = float(row[4])

            projection = proj(long, lat)
            x = projection[0]
            y = projection[1]
            output_row = [colour, x, y, variance]
            csv_writer.writerow(output_row)

            x_arr.append(x)
            y_arr.append(y)

    if plot:
        plt.scatter(x_arr, y_arr)
        plt.xlabel('X-Position')
        plt.ylabel('Y-Position')
        plt.show()
