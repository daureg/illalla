#! /usr/bin/env python
# vim: set fileencoding=utf-8
# pylint: disable=E1101
"""Use least square and Rank SVM to optimize features weights, based on
alleged ground truth."""
from collections import namedtuple, Counter
from subprocess import check_call
from timeit import default_timer as clock
import numpy as np
import prettyplotlib as ppl
import lru_cache
import os
import shlex
import ujson
import neighborhood as nb
import utils as u

Sample = namedtuple('Sample', 'region label city'.split())
City = namedtuple('City', 'features surroundings support'.split())

with open('static/cpresets.json') as infile:
    gold_list = ujson.load(infile)
labels = {str(name): idx
          for idx, name in enumerate(sorted(gold_list.iterkeys()))}
samples = []
for neighborhood, label in labels.iteritems():
    city = 'paris'
    region = gold_list[neighborhood]['geo']
    samples.append(Sample(region, label, city))
    for city, regions in gold_list[neighborhood]['gold'].iteritems():
        for geojson in regions:
            samples.append(Sample(geojson['geometry'], label, city))
cities = ['paris'] + gold_list[labels.keys()[0]]['gold'].keys()


def city_desc(name):
    venues_info = nb.cn.gather_info(name, knn=5)
    surroundings, _ = nb.load_surroundings(name)
    support = nb.features_support(venues_info['features'])
    return City(venues_info, surroundings, support)
cities_desc = {name: city_desc(name) for name in cities}


@u.memodict
def sample_distrib(idx):
    """Return features distribution of venues in `sample`."""
    sample = samples[idx]
    center, radius, _, contains = nb.polygon_to_local(sample.city,
                                                      sample.region)
    city = cities_desc[sample.city]
    query = nb.describe_region(center, radius, contains, city.surroundings,
                               city.features)
    features, _, weights, vids = query  # may use times as well
    return nb.features_as_density(features, weights, city.support)


@lru_cache.lru_cache(500)
def distance_vector(sample1, sample2):
    return [nb.jensen_shannon_divergence(p, q)
            for p, q in zip(sample_distrib(sample1), sample_distrib(sample2))]

pdist = np.zeros((len(samples)*(len(samples)-1)/2,
                  cities_desc['paris'].features['features'].shape[1]))
same_label = []
row = 0
for i in range(len(samples)):
    for j in range(i+1, len(samples)):
        pdist[row, :] = distance_vector(i, j)
        same_label.append(1 if samples[i].label == samples[j].label else -1)
        row += 1
theta = np.ones((31,)) / np.sqrt(31)
class_size = Counter(same_label)
sl = np.array(same_label).astype(float)
sl[sl > 0] = 1.0/class_size[1]
sl[sl < 0] = -1.0/class_size[1]
A = sl.reshape(len(same_label), 1)*pdist
np.linalg.norm(np.dot(A, theta))
from pymatbridge import Matlab
mlab = Matlab(matlab='/m/fs/software/matlab/r2014a/bin/glnxa64/MATLAB',
              maxtime=10)
mlab.start()
N = len(samples)
involved = [[] for _ in range(N)]
row = 0
for i in range(N):
    for j in range(i+1, N):
        involved[i].append(row)
        involved[j].append(row)
        row += 1
rows = set(range(row))
notinvolved = [list(rows.difference(_)) for _ in involved]
dsts = np.zeros((len(samples), 2))
thetas = np.zeros((len(samples), 31))
i = 0
for train_row, test_row in zip(notinvolved, involved):
    res = mlab.run_func('optimize_weights.m', {'A': A[train_row, :].tolist()})
    x = np.array(res['result'])
    dsts[i, :] = [np.linalg.norm(np.dot(A[test_row, :], x)),
                  np.linalg.norm(np.dot(A[test_row, :], theta))]
    thetas[i, :] = x.ravel()
    i += 1

s = clock()
res = mlab.run_func('optimize_weights.m',
                    {'A': A.tolist(), 'involved': involved,
                     'notinvolved': notinvolved})
clock() - s

mlab.stop()
dsts = np.array(res['result']['distances'])
thetas = np.array(res['result']['thetas'])
# TODO: find a better way to learn theta on the complete dataset
matlab = np.array([0.039559, 0.35603, 0.35603, 0.039559, 0.039559, 0.039559,
                   0.039559, 0.039559, 0.039559, 0.039559, 0.039559,
                   0.039559, 0.039559, 0.039559, 0.039559, 0.35603, 0.039559,
                   0.35603, 0.35603, 0.35603, 0.27692, 0.039559, 0.039559,
                   0.039559, 0.35603, 0.039559, 0.039559, 0.039559, 0.039559,
                   0.039559, 0.039559])
for t in thetas:
    ppl.plot(t)
ppl.plot(matlab, lw=2)

score = np.ones((len(labels), len(labels))) + np.diag(2*np.ones(6))
score[0, 1] = 2
score[1, 0] = 2
# import sklearn.preprocessing as pp
# ps = pp.StandardScaler()
# ndist = ps.fit_transform(pdist)
# B = sl.reshape(len(same_label), 1)*ndist


def svm_line(data):
    res = '{} qid:{:d} {}'
    features = data[2:]
    assert len(features) == 31
    sfeatures = ['{}:{:.6f}'.format(i+1, f) for i, f in enumerate(features)]
    return res.format(data[0], int(data[1]), ' '.join(sfeatures))
exe = os.path.expanduser('~/rank/svm_rank_learn')
N = len(samples)
svm_thetas = np.zeros((N, 31))
for hidden in range(N):
    learn = np.zeros(((N-2)*(N-1), 1 + 1 + 31))
    row = 0
    for qid in range(N):
        if qid == hidden:
            continue
        for sid in range(N):
            if sid in [qid, hidden]:
                continue
            learn[row, :] = [score[samples[qid].label, samples[sid].label],
                             qid] + distance_vector(qid, sid)
            row += 1
    with open('train_{}.dat'.format(hidden), 'w') as out:
        out.write('\n'.join(map(svm_line, learn)))
    cmd = '{} -c 10 train_{}.dat model_{}.dat'
    check_call(shlex.split(cmd.format(exe, hidden, hidden)))
    with open('model_{}.dat'.format(hidden)) as model:
        raw_vec = model.readlines()[-1].strip()
        vec = np.array([float(_.split(':')[1])
                        for _ in raw_vec[2:-2].split(' ')])
    vec = vec / np.linalg.norm(vec)
    print(np.linalg.norm(np.dot(A, vec)))
    svm_thetas[hidden, :] = vec

from scipy.interpolate import pchip
for t in svm_thetas:
    interp = pchip(np.arange(len(t)), t)
    xx = np.linspace(0, len(t)-1, 101)
    ppl.plot(xx, interp(xx))
