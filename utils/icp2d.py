# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# iterative closest point
# inspired by http://stackoverflow.com/questions/20120384/iterative-closest-point-icp-implementation-on-python

# <codecell>

import cv2
import numpy as np
import sys
from numpy.random import *
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from copy import deepcopy


def del_miss(indices, dist, max_dist, th_rate=0.8):
    """Given indices of the src that correspond to closest neighbor
    of each point in dst, filter bad correspondences using the max_dist"""
    output_indices = []
    for i, src_idx in enumerate(indices):
        if dist[i] < max_dist:
            output_indices.append([int(src_idx), i])
    return np.array(output_indices)


def is_converge(Tr, scale):
    delta_angle = 0.0001
    delta_scale = scale * 0.0001

    min_cos = 1 - delta_angle
    max_cos = 1 + delta_angle
    min_sin = -delta_angle
    max_sin = delta_angle
    min_move = -delta_scale
    max_move = delta_scale

    return min_cos < Tr[0, 0] and Tr[0, 0] < max_cos and \
           min_cos < Tr[1, 1] and Tr[1, 1] < max_cos and \
           min_sin < -Tr[1, 0] and -Tr[1, 0] < max_sin and \
           min_sin < Tr[0, 1] and Tr[0, 1] < max_sin and \
           min_move < Tr[0, 2] and Tr[0, 2] < max_move and \
           min_move < Tr[1, 2] and Tr[1, 2] < max_move


def icp(d1, d2, max_iterate=10000, dist_threshold=3, convergence_limit=1e-2):
    """Estimate the transform from d2 -> d1"""
    knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(d1)

    src = deepcopy(d1)
    dst = deepcopy(d2)

    plt.figure()
    plt.title("Pre-ICP")
    plt.scatter(d1[:, 0], d1[:, 1])
    plt.scatter(d2[:, 0], d2[:, 1])

    prev_dst = None
    T_cumulative = np.eye(3)
    for i in range(max_iterate):
        distances, indices = knn.kneighbors(dst)
        indices = del_miss(indices, distances, dist_threshold)

        T, inliers = cv2.estimateAffine2D(dst[indices[:, 1]], src[indices[:, 0]])
        dst_homo = np.concatenate([dst, np.zeros((dst.shape[0], 1))], axis=1)
        dst = (T @ dst_homo.T).T
        dst = dst[:, :2]

        if isinstance(prev_dst, type(None)):
            prev_dst = deepcopy(dst)
        else:
            delta_dst = np.abs(prev_dst-dst)
            if np.sum(delta_dst) < convergence_limit:
                break
            prev_dst = dst

        T = np.append(T, np.array([0, 0, 1]).reshape((1, 3)), axis=0)
        T_cumulative = T @ T_cumulative

    plt.figure()
    plt.title("Post-ICP")
    plt.scatter(src[:, 0], src[:, 1])
    plt.scatter(dst[:, 0], dst[:, 1])
    plt.show()

    return T_cumulative[:2, :]
