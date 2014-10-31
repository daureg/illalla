#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Distance function between two venues for ITML and t-SNE."""
import scipy.io as sio
import numpy as np
import scipy
import math

COV = sio.loadmat('ITMLall.mat')['A']
# pylint: disable=E1101
COV = np.insert(COV, 5, values=0, axis=1)
COV = np.insert(COV, 5, values=0, axis=0)
COV[5, 5] = 1.0
COV = np.linalg.inv(COV)


def dst_itml(u, v, _):
    return scipy.spatial.distance.mahalanobis(u, v, COV)


def dst_tsne(u, v, _):
    return math.sqrt((u[0] - v[0])*(u[0] - v[0]) + (u[1] - v[1])*(u[1] - v[1]))
