#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Plot dataset in (reduced) dimension 2 or 3."""
import sys
try:
    # old version of matplotlib on some computers
    del sys.path[sys.path.index('/usr/lib/pymodules/python2.7')]
except ValueError:
    pass
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import prettyplotlib as ppl
import matplotlib as mpl
from sklearn import manifold, decomposition  # , datasets
import calc_tsne


def plot_embedding(figure, index, method, run_time, data, classes, dimension):
    """Scatter subplot `data` with colors corresponding to `classes` on
    `figure` at position `index` in `dimension`D. Title is made of `method`
    and `run_time`."""
    common = dict(c=classes, cmap=mpl.cm.Spectral, alpha=0.85)
    if dimension == 2:
        axe = figure.add_subplot(5, 3, 1 + index)
        ppl.scatter(data[:, 0], data[:, 1], **common)
    elif dimension == 3:
        axe = figure.add_subplot(5, 3, 1 + index, projection="3d")
        ppl.scatter(data[:, 0], data[:, 1], data[:, 2], **common)
    else:
        raise ValueError(dimension)
    plt.title("{} ({:.2g} sec)".format(method, run_time))
    axe.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
    axe.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
    if dimension == 3:
        axe.zaxis.set_major_formatter(mpl.ticker.NullFormatter())
    plt.axis('tight')


def choose_decomposition_method(method, n_components):
    """Return the decomposition corresponding to `method`."""
    if method == 'PCA':
        return decomposition.PCA(n_components)
    elif method == 'Randomized PCA':
        return decomposition.RandomizedPCA(n_components)
    elif method == 'Kernel PCA':
        return decomposition.KernelPCA(n_components, kernel='rbf')
    elif method == 'Sparse PCA':
        return decomposition.SparsePCA(n_components, n_jobs=1)
    elif method == 'SVD':
        return decomposition.TruncatedSVD(n_components)
    elif method == 'Factor Analysis':
        return decomposition.FactorAnalysis(n_components)
    elif method == 'ICA':
        return decomposition.FastICA(n_components)
    raise ValueError('{} is not a known method'.format(method))


def choose_manifold_method(method, n_components, n_neighbors):
    """Return the manifold corresponding to `method`."""
    method = method.lower()
    if method in ['standard', 'ltsa', 'hessian', 'modified']:
        # solver = 'auto' if method != 'standard' else 'dense'
        return manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                               eigen_solver='dense',
                                               method=method)
    elif method == 'isomap':
        return manifold.Isomap(n_neighbors, n_components)
    elif method == 'mds':
        return manifold.MDS(n_components, max_iter=200, n_init=1)
    elif method == 'spectral':
        return manifold.SpectralEmbedding(n_components=n_components,
                                          n_neighbors=n_neighbors)
    elif method == 't-sne':
        return calc_tsne.tSNE(n_components)
    raise ValueError('{} is not a known method'.format(method))


def compute_embedding(high, method, n_components=2, n_neighbors=None):
    """Reduce dimension of `high` to `n_components` using `method`
    (parametrized by `n_neighbors`)"""
    n_neighbors = n_neighbors or (n_components * (n_components + 3) / 2) + 4
    try:
        projector = choose_manifold_method(method, n_components, n_neighbors)
    except ValueError:
        projector = choose_decomposition_method(method, n_components)
    start = time()
    lower = projector.fit_transform(high)
    return lower, time() - start

if __name__ == '__main__':
    from timeit import default_timer as time
    from ClosestNeighbor import load_matrix
    # pylint: disable=C0103
    city = sys.argv[1].strip().lower()
    nb_dim = 2 if len(sys.argv) <= 2 else int(sys.argv[2])
    features = load_matrix(city)['v']
    features[:, 5] = features[:, 5] / 8e5
    cats = (8*features[:, 5]).astype(int)
    Axes3D
    # n_points = 300
    # X, color = datasets.samples_generator.make_s_curve(n_points,
    # features, cats = datasets.samples_generator.make_swiss_roll(n_points,
    #                                                             noise=0.1,
    #                                                           random_state=0)
    fig = plt.figure(figsize=(34, 38))
    title = "{} venues of {} projected to {} dimensions"
    title = title.format(features.shape[0], city.title(), nb_dim)
    print(title)
    plt.suptitle(title, fontsize=14)
    # ax = fig.add_subplot(241, projection='3d')
    # plt.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=mpl.cm.Spectral)

    methods = ['Standard', 'LTSA', 'Hessian', 'Modified', 'Isomap', 'MDS',
               'Spectral', 't-SNE', 'PCA', 'Randomized PCA', 'Kernel PCA',
               'Sparse PCA', 'SVD', 'Factor Analysis', 'ICA']

    for i, method in enumerate(methods):
        reduced, how_long = compute_embedding(features, method, nb_dim)
        plot_embedding(fig, i, method, how_long, reduced, cats, nb_dim)
        print("{}: {:.2g} sec".format(method, how_long))
    outfile = '{}_DR_{}.png'.format(city, nb_dim)
    plt.savefig(outfile, frameon=False, bbox_inches='tight',
                pad_inches=0.05)
