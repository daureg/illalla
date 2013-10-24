#! /usr/bin/python
# vim: set fileencoding=utf-8
import numpy as np
import scipy.io as sio
from timeit import default_timer as clock
from scipy.spatial.distance import pdist, squareform
GREAT = 50

d = sio.loadmat('points.mat')
p = d['points']
start = clock()
grav = np.mean(p, 0)
tmp = p - grav
dst = np.sum(tmp**2, 1)
print('{:.5f}s'.format(clock() - start))
print(np.mean(dst))
print(np.std(dst))

start = clock()
dst = pdist(p)
print('{:.5f}s'.format(clock() - start))
print(np.mean(dst))
print(np.std(dst))

pd = squareform(dst)
np.fill_diagonal(pd, GREAT)
dst = np.min(pd, 1)
print('{:.5f}s'.format(clock() - start))
print(np.mean(dst))
print(np.std(dst))
