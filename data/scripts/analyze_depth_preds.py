import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt

path = '/media/carter/Samsung_T5/runs/leonhard_runs_split/depth_stats.pkl'
with open(path, 'rb') as pickle_f:
    data = pickle.load(pickle_f)

data['depth_error'] = np.asarray(data['depth_error'])
data['depth_true'] = np.asarray(data['depth_true'])

range_arr = []
err_arr = []
for range_min in range(0, 60, 5):
    mask = np.logical_and(data['depth_true'] >= range_min,
                          data['depth_true'] < range_min + 5)
    mean_error = np.median(data['depth_error'][mask])
    range_arr.append(range_min)
    err_arr.append(mean_error)

plt.title('Range vs Median Error')
plt.xlabel('Range (m)')
plt.ylabel('Error (m')
plt.scatter(range_arr, err_arr)
plt.show()
print('hi')