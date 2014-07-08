#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Try to find low EMD distance regions fast."""
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from warnings import warn
import itertools
import matplotlib as mpl
import prettyplotlib as ppl
import matplotlib.pyplot as plt
import neighborhood as nb
import numpy as np
import persistent as p
import ujson
from shapely.geometry import Polygon, Point
from timeit import default_timer as clock

# load data
with open('static/ground_truth.json') as infile:
    gold_list = ujson.load(infile)
districts = sorted(gold_list.iterkeys())
cities = sorted(gold_list[districts[0]]['gold'].keys())
cities_desc = {name: nb.cn.gather_info(name, raw_features=True,
                                       hide_category=True)
               for name in cities}
WHICH_GEO = []


def test_all_queries(queries, query_city='paris'):
    all_res = []
    timing = []
    for query in queries:
        target_city, district = query
        possible_regions = gold_list[district]['gold'].get(query_city)
        region = nb.choose_query_region(possible_regions)
        if not region:
            all_res.append([])
            continue
        start = clock()
        infos = nb.interpret_query(query_city, target_city, region, 'emd')
        _, right, _, regions_distance, _, threshold = infos
        print(query, threshold)

        vloc = cities_venues[target_city]
        infos = retrieve_closest_venues(district, query_city, target_city)
        candidates, _, _ = infos
        eps, mpts = 210, 18 if len(vloc) < 5000 else 50
        clusters = good_clustering(vloc, list(sorted(candidates)), eps, mpts)
        res = []
        for cluster in clusters:
            venues_areas = cluster_to_venues(cluster, vloc)
            if len(venues_areas) == 0:
                continue
            for venues in venues_areas:
                venues = right['features'][venues, :]
                res.append(regions_distance(venues.tolist(),
                                            nb.weighting_venues(venues[:, 1])))
        WHICH_GEO.append(np.argmin(res) % len(venues_areas))
        all_res.append(res)
        timing.append(clock() - start)
    return all_res, timing


def cluster_to_venues(indices, vloc):
    # Given a cluster (ie a set of venues indices), it should return
    # neighborhoods (ie compact/complete sets of venues indices) that will be
    # evaluated by EMD.
    # Given how DBSCAN works, most of these clusters look rather convex, so
    # convex hull could be a good option. Otherwise, I could use CGAL binding
    # to get alpha shapes. Then I can consider bounding box (called envelope
    # by Shapely) or circle. Finally, some dilation and erosion of the
    # previous shapes.
    # I can also add or remove individual points (but it's unclear which one,
    # see notebook) while maintaining more or less neighborhood property.
    # Also given, these shapes, it would be nice to retrieve corresponding
    # venues fast, given that at this point, I know only pairwise distance of
    # candidates venues (so maybe I can build a cKDTree, query a big a circle
    # around the cluster and test individually venues to see if they belong to
    # the considered shape)
    points = vloc[indices, :]
    try:
        hull = points[ConvexHull(points).vertices, :]
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        print(indices)
        return []
    poly = Polygon(hull)
    box = poly.envelope
    return [[idx for idx, loc in enumerate(vloc)
             if region.intersects(Point(loc))]
            for region in [poly, box]]


def get_candidates_venues(query_features, target_features):
    """Return the set of all 50 closest venues from `query_features` to
    `target_features`."""
    distances = cdist(query_features, target_features)
    ordered = np.argsort(distances, 1)
    return set(ordered[:, :50].ravel())


def retrieve_closest_venues(district, query_city, target_city):
    """For the given query, return a list of venues indices for knn level of
    50, as well as a list of indices for each gold area and the threshold
    number of venues."""
    gold = gold_list[district]['gold']
    query = gold[query_city][0]
    query_venues = query['properties']['venues']
    mask = np.where(np.in1d(cities_desc[query_city]['index'], query_venues))[0]
    query_features = cities_desc[query_city]['features'][mask, :]
    all_target_features = cities_desc[target_city]['features']
    tindex = cities_desc[target_city]['index']
    gold_venue_indices = [np.where(np.in1d(tindex,
                                           reg['properties']['venues']))[0]
                          for reg in gold[target_city]
                          if len(reg['properties']['venues']) >= 20]
    if not gold_venue_indices:
        msg = '{} in {} has no area with at least 20 venues'
        warn(msg.format(district, target_city.title()))
        # return None, None, None
    candidates = get_candidates_venues(query_features, all_target_features)
    threshold = int(len(tindex)*1.0*len(query_venues) /
                    len(cities_desc[query_city]['index']))
    return candidates, gold_venue_indices, threshold


def f_score(recall, precision, beta=2.0):
    return (1+beta*beta)*(recall * precision)/(beta*beta*precision + recall)


# load venues location for all cities
cities_venues_raw = {name: p.load_var(name+'_svenues.my') for name in cities}
cities_venues = {}
cities_index = {}
for city in cities:
    vids, _, locs = cities_venues_raw[city].all()
    vindex = cities_desc[city]['index']
    cities_venues[city] = np.zeros((len(vindex), 2))
    cities_index[city] = dict(itertools.imap(lambda x: (x[1], x[0]),
                                             enumerate(vindex)))
    for vid, loc in itertools.izip(vids, locs):
        pos = cities_index[city].get(vid)
        if pos is not None:
            cities_venues[city][pos, :] = loc

gray = '#bdbdbd'
red = '#e51c23'
green = '#64dd17'
blue = '#03a9f4'
orange = '#f57c00'


def evaluate_clustering(labels, candidates_indices, gold_indices_list):
    fscores = []
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    for k in range(n_clusters):
        best_score = np.nan
        for idx, tg in enumerate(gold_indices_list):
            relevant = np.sum(np.in1d(candidates_indices[labels == k], tg))
            precision = relevant*1.0 / candidates_indices[labels == k].size
            recall = relevant*1.0 / len(tg)
            fscore = f_score(recall, precision, beta=1.0)
            if not np.isnan(fscore):
                if np.isnan(best_score):
                    best_score = fscore
                else:
                    best_score = max(fscore, best_score)
            # fscores.append(fscore)
        fscores.append(best_score)
    assert len(fscores) == n_clusters
    # mean of F1-score of best gold, 0 if nan (ie precision = 0)
    return [np.mean(np.nan_to_num(fscores)), n_clusters]


QUERIES = itertools.product(cities, districts)
ALL_Q = [(city, district) for city, district in QUERIES
         if city not in ['paris', 'berlin'] and
         city in gold_list[district]['gold'] and
         [1 for reg in gold_list[district]['gold'][city]
          if len(reg['properties']['venues']) >= 20]]


def cluster_is_small_enough(max_length, max_venues, vloc):
    """Make sure than `vlocs` is within acceptable constraints in terms of space
    and number of venues."""
    if len(vloc) > max_venues:
        return False
    dim_x, dim_y = [vloc[:, _].max() - vloc[:, _].min() for _ in [0, 1]]
    return all([dim <= max_length for dim in [dim_x, dim_y]])


def good_clustering(locs, cands, eps, mpts):
    """Return a list of list of indices making up clusters of acceptable
    size."""
    clocs = locs[cands, :]
    pwd = squareform(pdist(clocs))
    clusters_indices = recurse_dbscan(pwd, np.arange(len(cands)), clocs,
                                      eps, mpts)
    if not clusters_indices:
        clusters_indices = recurse_dbscan(pwd, np.arange(len(cands)), clocs,
                                          eps*1.3, mpts/1.4)
    cands = np.array(cands)
    return [cands[c] for c in clusters_indices]


def recurse_dbscan(distances, indices, locs, eps, mpts, depth=0):
    """Do a first DBSCAN with given parameters and if some clusters are too
    big, recluster them using stricter parameters."""
    # msg = '{}Cluster {} points with ({}, {})'
    # instead http://stackoverflow.com/a/24308860
    # print(msg.format(depth*'\t', len(indices), eps, mpts))
    pwd = distances
    mpts = int(mpts)
    labels = DBSCAN(eps=eps, min_samples=int(mpts),
                    metric='precomputed').fit(pwd).labels_
    cl_list = []
    for k in np.unique(labels):
        if k == -1:
            continue
        k_indices = np.argwhere(labels == k).ravel()
        if cluster_is_small_enough(1.3e3, 230, locs[k_indices, :]):
            # msg = '{}add one cluster of size {}'
            # print(msg.format(depth*'\t'+'  ', len(k_indices)))
            cl_list.append(indices[k_indices])
        else:
            if depth < 3:
                sub_pwd = pwd[np.ix_(k_indices, k_indices)]
                sub_locs = locs[k_indices, :]
                sub_indices = recurse_dbscan(sub_pwd, k_indices, sub_locs,
                                             eps/1.4, mpts*1.3, depth+1)
                cl_list.extend([indices[c] for c in sub_indices])
            else:
                warn('Cannot break one cluster at level {}'.format(depth))
    return cl_list


def plot_clusters(clusters, candidates, bounds, vloc, hulls, shrink=0.9):
    """Plot all `clusters` among `candidates` with the `bounds` of the city
    (or at least `shrink` of them). Also plot convex `hulls` of gold areas if
    provided."""
    xbounds, ybounds = bounds
    unique_labels = len(clusters)
    clustered = set().union(*map(list, clusters))
    noise = list(candidates.difference(clustered))
    if unique_labels > 5:
        colors = mpl.cm.Spectral(np.linspace(0, 1, unique_labels+1))
    else:
        colors = [gray, red, green, blue, orange]
    plt.figure(figsize=(20, 15))
    for k, indices, col in zip(range(unique_labels+1), [noise]+clusters,
                               colors):
        k -= 1
        if k == -1:
            col = 'gray'
        ppl.scatter(vloc[indices, 0], vloc[indices, 1],
                    s=35 if k != -1 else 16, color=col,
                    alpha=0.8 if k != -1 else 0.6,
                    label='noise' if k == -1 else 'cluster {}'.format(k+1))
    hulls = hulls or []
    for idx, hull in enumerate(hulls):
        first_again = range(len(hull))+[0]
        ppl.plot(hull[first_again, 0], hull[first_again, 1], '--',
                 c=ppl.colors.almost_black, lw=1.0, alpha=0.9,
                 label='gold region' if idx == 0 else None)
    plt.xlim(shrink*xbounds)
    plt.ylim(shrink*ybounds)
    ppl.legend()

if __name__ == '__main__':
    import sys
    sys.exit()
    query_city, target_city, district = 'paris', 'barcelona', 'triangle'
    vloc = cities_venues[target_city]
    xbounds = np.array([vloc[:, 0].min(), vloc[:, 0].max()])
    ybounds = np.array([vloc[:, 1].min(), vloc[:, 1].max()])
    infos = retrieve_closest_venues(district, query_city, target_city)
    top_venues, gold_venues_indices, threshold = infos
    gold_venues = set().union(*map(list, gold_venues_indices))
    candidates = top_venues
    hulls = [vloc[tg, :][ConvexHull(vloc[tg, :]).vertices, :]
             for tg in gold_venues_indices]
    eps, mpts = 210, 18
    sclidx = good_clustering(vloc, list(sorted(candidates)), eps, mpts)
    print(np.array(map(len, sclidx)))
