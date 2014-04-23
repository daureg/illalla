#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Plot time pattern of all cities."""
import matplotlib
# matplotlib.use('Agg')
import prettyplotlib as ppl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import sklearn.metrics as skm
import utils as u
from ClosestNeighbor import FEATURES
TRIVIAL = [(1, 2), (0, 1), (0, 2), (0, 3), (2, 3)]


def plotc(f, i, j, r, m, mi=False):
    """Plot f(i) against f(j) as column"""
    plt.figure()
    ppl.plot(f[:, i], f[:, j], 'r+')
    plt.xlabel(FEATURES[i])
    plt.ylabel(FEATURES[j])
    m = 'm_i  = {:.3f}'.format(m)
    r = 'r  = {:.3f}'.format(r)
    order = [m, r] if mi else [r, m]
    plt.title('{} ({})'.format(*order))


def mi(x, y, bins=10):
    """Mutual information between x and y"""
    H_x = u.compute_entropy(np.histogram(x, bins)[0])
    H_y = u.compute_entropy(np.histogram(y, bins)[0])
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = skm.mutual_info_score(None, None, contingency=c_xy)
    return mi/np.sqrt(H_x*H_y)


def all_mi(mat, bins=10):
    """Pairwise mutual information between columns of `mat`"""
    dim = mat.shape[1]
    res = np.zeros((dim, dim))
    for i in range(dim):
        res[i, i+1:] = [mi(mat[:, i], mat[:, j], bins)
                        for j in range(i+1, dim)]
    return res


# https://stackoverflow.com/a/2690063
def pairs(name, data, labels=None):
    """ Generate something similar to R `pairs` """
    nvariables = data.shape[1]
    mpl.rcParams['figure.figsize'] = 3.5*nvariables, 3.5*nvariables
    if labels is None:
        labels = ['var {}'.format(i) for i in range(nvariables)]
    fig = plt.figure()
    s = clock()
    for i in range(nvariables):
        for j in range(i, nvariables):
            nsub = i * nvariables + j + 1
            ax = fig.add_subplot(nvariables, nvariables, nsub)
            ax.tick_params(left='off', bottom='off', right='off', top='off',
                           labelbottom='off', labelleft='off')
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['right'].set_visible(False)
            if i == j:
                ppl.hist(ax, data[:, i], grid='y')
                ax.set_title(labels[i], fontsize=10)
            else:
                ax.set_xlim([data[:, i].min(), data[:, i].max()])
                ax.set_ylim([data[:, j].min(), data[:, j].max()])
                ax.scatter(data[:, i], data[:, j], marker='.', color='k', s=4)
            ax.tick_params(labelbottom='off', labelleft='off')
    print(clock() -s)
    s = clock()
    plt.savefig(name+'_corr.png', dpi=96, transparent=False, frameon=False,
                bbox_inches='tight', pad_inches=0.1)
    print(clock() -s)


def load_cities(verbose=True):
    """load all available matrices"""
    import os
    res = {}
    for mats in os.listdir('.'):
        if not mats.endswith('_fv.mat'):
            continue
        name, mat = mats.split('_')[0], sio.loadmat(mats)['v']
        weird = np.logical_or(np.isinf(mat), np.isnan(mat))
        mat[weird] = 0.0
        res[name] = mat

    if verbose:
        print('\n'.join(['{}: {} venues'.format(k, v.shape[0])
                         for k, v in res.iteritems()]))
    return res


def show_dependencies(all_features, city):
    "plot features with high correlation"""
    m = all_mi(all_features[city])
    c = np.corrcoef(all_features[city].T, ddof=1)
    top_r = np.triu(np.logical_and(np.abs(c) > 0.5, c < 1))
    high_r = sorted(np.argwhere(top_r), key=lambda x: c[x[0], x[1]],
                    reverse=True)
    top_m = list(reversed(zip(*np.unravel_index(np.argsort(m.ravel())[-10:],
                                                m.shape))))
    high_m = [idx for idx in top_m if m[idx[0], idx[1]] > 0.2]
    showed = TRIVIAL[:]
    for i in high_r:
        if tuple(i) not in TRIVIAL:
            showed.append(tuple(i))
            plotc(all_features[city], i[0], i[1],
                  c[i[0], i[1]], m[i[0], i[1]])
    for i in high_m:
        if i not in showed:
            plotc(all_features[city], i[0], i[1],
                  c[i[0], i[1]], m[i[0], i[1]], True)

from timeit import default_timer as clock
if __name__ == '__main__':
    np.set_printoptions(linewidth=230,
                        formatter={'float_kind': '{:.3f}'.format})
    af = load_cities()
    # city = 'barcelona'
    # columns = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    #            19, 20, 21, 22, 23]
    # ncat = af[city]
    # ncat[:, 5] = ncat[:, 5] / 1e5
    # pairs(city, ncat[:, columns], [FEATURES[_] for _ in columns])
    # show_dependencies(city)
    # s = clock()
    # r=np.random.random((40,4))
    # print(clock() -s)
    # pairs('r', r)
    # print(clock() -s)
