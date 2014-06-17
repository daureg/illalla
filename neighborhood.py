#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Match polygonal regions between cities
Input:
    name of two cities
    a list of coordinates that make up a polygon in one city
Output:
    a list of coordinates that make up a polygon in the other city
"""
from __future__ import print_function
import cities
import ClosestNeighbor as cn
# import explore as xp
import numpy as np
import utils as u
import itertools as i
import shapely.geometry as sgeo
import scipy.cluster.vq as vq
import emd_leftover
# pylint: disable=E1101
# pylint: disable=W0621
NB_CLUSTERS = 3
JUST_READING = False
MAX_EMD_POINTS = 600
QUERY_NAME = None
MATLAB_PATH = '/m/fs/software/matlab/r2014a/bin/glnxa64/MATLAB'
MATLAB = None


def profile(func):
    return func


@profile
def load_surroundings(city):
    """Load projected coordinates and extra field of all venues, checkins and
    photos within `city`, as well as returning city geographical bounds."""
    import persistent as p
    surroundings = [p.load_var('{}_svenues.my'.format(city)), None, None]
    # surroundings = [p.load_var('{}_s{}s.my'.format(city, kind))
    #                 for kind in ['venue', 'checkin', 'photo']]
    venues_pos = np.vstack(surroundings[0].loc)
    city_extent = list(np.min(venues_pos, 0)) + list(np.max(venues_pos, 0))
    return surroundings, city_extent


@profile
def polygon_to_local(city, geojson):
    """Convert a `geojson` geometry to a local bounding box in `city`, and
    return its center, radius and a predicate indicating membership."""
    assert geojson['type'] == 'Polygon'
    coords = np.fliplr(np.array(geojson['coordinates'][0]))
    projected = sgeo.Polygon(cities.GEO_TO_2D[city](coords))
    minx, miny, maxx, maxy = projected.bounds
    center = list(projected.centroid.coords[0])
    radius = max(maxx - minx, maxy - miny)*0.5
    return center, radius, projected.bounds, projected.contains


@profile
def describe_region(center, radius, belongs_to, surroundings, city_fv,
                    threshold=10):
    """Return the description (X, x, w, ids) of the region defined by
    `center`, `radius` and `belongs_to`, provided that it contains enough
    venues."""
    svenues, scheckins, sphotos = surroundings
    vids, _ = gather_entities(svenues, center, radius, belongs_to,
                              threshold)
    if not vids:
        return None, None, None, None
    vids = filter(city_fv['index'].__contains__, vids)
    if len(vids) < threshold:
        return None, None, None, None
    # _, ctime = gather_entities(scheckins, center, radius, belongs_to)
    # _, ptime = gather_entities(sphotos, center, radius, belongs_to)
    mask = np.where(np.in1d(city_fv['index'], vids))[0]
    assert mask.size == len(vids)
    weights = weighting_venues(city_fv['features'][mask, 1])
    # time_activity = lambda visits: xp.aggregate_visits(visits, 1, 4)[0]
    # activities = np.hstack([xp.to_frequency(time_activity(ctime)),
    #                         xp.to_frequency(time_activity(ptime))])
    activities = np.ones((12, 1))
    return city_fv['features'][mask, :], activities, weights, vids


@profile
def features_support(features):
    """Return a list of intervals representing the support of the probability
    distribution for each dimension."""
    return zip(np.min(features, 0), np.max(features, 0))


@u.memodict
def right_bins(dim):
    extent = RIGHT_SUPPORT[dim][1] - RIGHT_SUPPORT[dim][0]
    bins = 10
    size = 1.0/bins
    return [RIGHT_SUPPORT[dim][0] + j*size*extent for j in range(bins+1)]


@profile
def features_as_density(features, weights, support, bins=10):
    """Turn raw `features` into probability distribution over each dimension,
    with respect to `weights`."""
    def get_bins_full(dim):
        extent = support[dim][1] - support[dim][0]
        size = 1.0/bins
        return [support[dim][0] + j*size*extent for j in range(bins+1)]

    get_bins = right_bins if support is RIGHT_SUPPORT else get_bins_full
    return np.vstack([np.histogram(features[:, i], weights=weights,
                                   bins=get_bins(i))[0]
                      for i in range(features.shape[1])])


def features_as_lists(features):
    """Turn numpy `features` into a list of list, suitable for emd
    function."""
    return features.tolist()


@profile
def weighting_venues(values):
    """Transform `values` into a list of positive weights that sum up to 1."""
    from sklearn.preprocessing import MinMaxScaler
    scale = MinMaxScaler()
    size = values.size
    scaled = scale.fit_transform(np.power(values, .2).reshape((size, 1)))
    normalized = scaled.ravel()/np.sum(scaled)
    normalized[normalized < 1e-6] = 1e-6
    return normalized


@profile
def gather_entities(surrounding, center, radius, belongs_to, threshold=0):
    """Filter points in `surrounding` that belong to the given region."""
    ids, info, locs = surrounding.around(center, radius)
    info = len(ids)*[0, ] if len(info) == 0 else list(info[0])
    if len(ids) < threshold:
        return None, None
    if belongs_to is None:
        return ids, info
    is_inside = lambda t: belongs_to(sgeo.Point(t[2]))
    res = zip(*(i.ifilter(is_inside, i.izip(ids, info, locs))))
    if len(res) != 3:
        return None, None
    ids[:], info[:], locs[:] = res
    if len(ids) < threshold:
        return None, None
    return ids, info


@profile
def jensen_shannon_divergence(P, Q):
    """Compute JSD(P || Q) as defined in
    https://en.wikipedia.org/wiki/Jensen–Shannon_divergence """
    avg = 0.5*(P + Q)
    avg_entropy = 0.5*(u.compute_entropy(P) + u.compute_entropy(Q))
    return u.compute_entropy(avg) - avg_entropy


@profile
def proba_distance(density1, global1, density2, global2, theta):
    """Compute total distances between all distributions"""
    proba = np.dot(theta, [jensen_shannon_divergence(p, q)
                           for p, q in zip(density1, density2)])
    return proba[0] + np.linalg.norm(global1 - global2)


SURROUNDINGS, CITY_FEATURES, THRESHOLD = None, None, None
METRIC_NAME, CITY_SUPPORT, DISTANCE_FUNCTION, RADIUS = None, None, None, None
RIGHT_SUPPORT = None


@profile
def one_cell(args):
    cx, cy, id_x, id_y, id_ = args
    center = [cx, cy]
    contains = None
    candidate = describe_region(center, RADIUS, contains,
                                SURROUNDINGS, CITY_FEATURES,
                                THRESHOLD)
    features, c_times, weights, c_vids = candidate
    if features is not None:
        if 'emd' in METRIC_NAME:
            c_density = features_as_lists(features)
            supp = weights
        elif 'cluster' == METRIC_NAME:
            c_density = features
            supp = weights
        elif 'leftover' in METRIC_NAME:
            c_density = features
            supp = (weights, id_)
        else:
            c_density = features_as_density(features, weights,
                                            CITY_SUPPORT)
            supp = c_times
        distance = DISTANCE_FUNCTION(c_density, supp)
        return [cx, cy, distance, c_vids]
    else:
        return [None, None, None, None]


@profile
def brute_search(city_desc, hsize, distance_function, threshold,
                 metric='jsd'):
    """Move a sliding circle over the whole city and keep track of the best
    result."""
    global SURROUNDINGS, CITY_FEATURES, THRESHOLD, RADIUS
    global METRIC_NAME, CITY_SUPPORT, DISTANCE_FUNCTION
    import multiprocessing
    RADIUS = hsize
    THRESHOLD = threshold
    METRIC_NAME = metric
    city_size, CITY_SUPPORT, CITY_FEATURES, city_infos = city_desc
    SURROUNDINGS, bounds = city_infos
    DISTANCE_FUNCTION = distance_function
    minx, miny, maxx, maxy = bounds
    nb_x_step = int(4*np.floor(city_size[0]) / hsize + 1)
    nb_y_step = int(4*np.floor(city_size[1]) / hsize + 1)
    best = [1e20, [], [], RADIUS]
    res_map = []
    pool = multiprocessing.Pool(4)

    x_steps = np.linspace(minx+hsize, maxx-hsize, nb_x_step)
    y_steps = np.linspace(miny+hsize, maxy-hsize, nb_y_step)
    x_vals, y_vals = np.meshgrid(x_steps, y_steps)
    to_cell_arg = lambda _: (float(_[1][0]), float(_[1][1]), _[0] % nb_x_step,
                             _[0]/nb_x_step, _[0])
    cells = i.imap(to_cell_arg, enumerate(i.izip(np.nditer(x_vals),
                                                 np.nditer(y_vals))))
    res = pool.map(one_cell, cells)
    res_map = []
    if MATLAB:
        MATLAB.start()
        dsts = emd_leftover.collect_matlab_output(len(res), MATLAB)
        for cell, dst in i.izip(res, dsts):
            if cell[0]:
                cell[2] = dst
        MATLAB.stop()
    for cell in res:
        if cell[0] is None:
            continue
        res_map.append(cell[:3])
        if cell[2] < best[0]:
            best = [cell[2], cell[3], [cell[0], cell[1]], RADIUS]

    if QUERY_NAME:
        import persistent as p
        p.save_var('comparaison/'+QUERY_NAME,
                   [[cell[2], len(cell[3]), [cell[0], cell[1]], RADIUS]
                    for cell in res if cell[0]])
    yield best, res_map, 1.0


def interpret_query(from_city, to_city, region, metric):
    """Load informations about cities and compute useful quantities."""
    # Load info of the first city
    left = cn.gather_info(from_city, knn=1)
    left_infos = load_surroundings(from_city)
    left_support = features_support(left['features'])

    # Compute info about the query region
    center, radius, _, contains = polygon_to_local(from_city, region)
    query = describe_region(center, radius, contains, left_infos[0], left)
    features, times, weights, vids = query
    print('{} venues in query region.'.format(len(vids)))
    venue_proportion = 1.0*len(vids) / left['features'].shape[0]

    # And use them to define the metric that will be used
    theta = np.ones((1, left['features'].shape[1]))
    theta = np.array([[0.039559, 0.35603, 0.35603, 0.039559, 0.039559,
                       0.039559, 0.039559, 0.039559, 0.039559, 0.039559,
                       0.039559, 0.039559, 0.039559, 0.039559, 0.039559,
                       0.35603, 0.039559, 0.35603, 0.35603, 0.35603, 0.27692,
                       0.039559, 0.039559, 0.039559, 0.35603, 0.039559,
                       0.039559, 0.039559, 0.039559, 0.039559,
                       0.039559]])
    ltheta = len(theta.ravel())*[1, ]

    if 'emd' in metric:
        from emd import emd
        from emd_dst import dist_for_emd
        query_num = features_as_lists(features)

        @profile
        def regions_distance(r_features, r_weigths):
            if len(r_features) >= MAX_EMD_POINTS:
                return 1e20
            return emd((query_num, map(float, weights)),
                       (r_features, map(float, r_weigths)),
                       lambda a, b: float(dist_for_emd(a, b, ltheta)))
    elif 'cluster' in metric:
        from scipy.spatial.distance import cdist
        query_num = weighted_clusters(features, NB_CLUSTERS, weights)

        def regions_distance(r_features, r_weigths):
            r_cluster = weighted_clusters(r_features, NB_CLUSTERS, r_weigths)
            costs = cdist(query_num, r_cluster).tolist()
            return min_cost(costs)
    elif 'leftover' in metric:
        from pymatbridge import Matlab
        global MATLAB
        MATLAB = Matlab(matlab=MATLAB_PATH, maxtime=30)

        @profile
        def regions_distance(r_features, second_arg):
            r_weigths, idx = second_arg
            emd_leftover.write_matlab_problem(features, weights, r_features,
                                              r_weigths, idx)
            return -1
    else:
        query_num = features_as_density(features, weights, left_support)

        @profile
        def regions_distance(r_density, r_global):
            """Return distance of a region from `query_num`."""
            return proba_distance(query_num, times, r_density, r_global,
                                  theta)

    # Load info of the target city
    right = cn.gather_info(to_city, knn=2)
    right_infos = load_surroundings(to_city)
    minx, miny, maxx, maxy = right_infos[1]
    right_city_size = (maxx - minx, maxy - miny)
    right_support = features_support(right['features'])
    global RIGHT_SUPPORT
    RIGHT_SUPPORT = right_support

    # given extents, compute treshold of candidate
    threshold = 0.7 * venue_proportion * right['features'].shape[0]
    right_desc = [right_city_size, right_support, right, right_infos]

    return [left, right, right_desc, regions_distance, vids, threshold]


def best_match(from_city, to_city, region, tradius, progressive=False,
               metric='jsd'):
    """Try to match a `region` from `from_city` to `to_city`. If progressive,
    yield intermediate result."""
    assert metric in ['jsd', 'emd', 'jsd-nospace', 'jsd-greedy', 'cluster',
                      'leftover']

    infos = interpret_query(from_city, to_city, region, metric)
    left, right, right_desc, regions_distance, vids, threshold = infos
    if JUST_READING:
        yield len(vids), None, None
        raise Exception()

    res, vals = None, None
    if metric.endswith('-nospace'):
        res, vals = search_no_space(vids, 10.0/7*threshold, regions_distance,
                                    left, right, RIGHT_SUPPORT)
    elif metric.endswith('-greedy'):
        res, vals = greedy_search(10.0/7*threshold, regions_distance, right,
                                  RIGHT_SUPPORT)
    else:
        # Use case for https://docs.python.org/3/whatsnew/3.3.html#pep-380
        for res, vals, progress in brute_search(right_desc, tradius,
                                                regions_distance, threshold,
                                                metric=metric):
            if progressive:
                yield res, vals, progress
            else:
                print(progress, end='\t')
    yield res, vals, 1.0


@profile
def weighted_clusters(venues, k, weights):
    """Return `k` centroids from `venues` (clustering is unweighted by
    centroid computation honors `weights` of each venues)."""
    labels = np.zeros(venues.shape[0])
    if k > 1:
        nb_tries = 0
        while len(np.unique(labels)) != k and nb_tries < 5:
            _, labels = vq.kmeans2(venues, k, iter=5, minit='points')
            nb_tries += 1
    try:
        return np.array([np.average(venues[labels == i, :], 0,
                                    weights[labels == i])
                         for i in range(k)])
    except ZeroDivisionError:
        print(labels)
        print(weights)
        print(np.sum(weights))
        raise


@profile
def min_cost(costs):
    """Return average min-cost of assignment of row and column of the `costs`
    matrix."""
    import munkres
    assignment = munkres.Munkres().compute(costs)
    cost = sum([costs[r][c] for r, c in assignment])
    return cost/len(costs)


def one_method_seed_regions(from_city, to_city, region, metric,
                            candidate_generation, clustering):
    """Return promising clusters matching `region`."""
    assert candidate_generation in ['knn', 'dst']
    assert clustering in ['discrepancy', 'dbscan']
    infos = interpret_query(from_city, to_city, region, metric)
    left, right, right_desc, regions_distance, vids, threshold = infos
    if candidate_generation == 'knn':
        candidates = get_knn_candidates(vids, left, right, threshold,
                                        at_most=15*threshold)
    elif candidate_generation == 'dst':
        candidates = get_neighborhood_candidates(regions_distance, right,
                                                 metric, at_most=15*threshold)

    clusters = find_promising_seeds(candidates[1], right_desc[3][0][0],
                                    clustering, right)
    how_many = min(len(clusters), 6)
    msg = 'size of cluster: '
    msg += str([len(_[1]) for _ in clusters])
    msg += '\ndistance, radius, nb_venues:\n'
    print(msg)
    for cluster in clusters[:how_many]:
        mask = np.where(np.in1d(right['index'], cluster[1]+cluster[2]))[0]
        weights = weighting_venues(right['features'][mask, 1])
        activities = np.ones((12, 1))
        features = right['features'][mask, :]
        if 'jsd' in metric:
            density = features_as_density(features, weights, right_desc[1])
            supplemental = activities
        elif 'emd' in metric:
            density = features_as_lists(features)
            supplemental = weights
        elif 'cluster' in metric:
            density = features
            supplemental = weights
        dst = regions_distance(density, supplemental)
        msg += '{:.4f}, {:.1f}, {}\n'.format(dst, np.sqrt(cluster[0].area),
                                             len(mask))
    print(msg)
    return [_[1] for _ in clusters[:how_many]], msg


def get_seed_regions(from_city, to_city, region):
    for metric in ['jsd', 'emd']:
        infos = interpret_query(from_city, to_city, region, metric)
        left, right, right_desc, regions_distance, vids, threshold = infos
        knn_cds = get_knn_candidates(vids, left, right, threshold, at_most=250)
        ngh_cds = get_neighborhood_candidates(regions_distance, right, metric,
                                              at_most=250)
        for _, candidates in [knn_cds, ngh_cds]:
            for scan in ['dbscan', 'discrepancy']:
                clusters = find_promising_seeds(candidates,
                                                right_desc[3][0][0], scan,
                                                right)
                for cl in clusters:
                    print(metric, scan, cl[1])


@profile
def greedy_search(nb_venues, distance_function, right_knn, support):
    """Find `nb_venues` in `right_knn` that optimize the total distance
    according to `distance_function`."""
    import random as r
    candidates_idx = []
    nb_venues = int(nb_venues)+3
    while len(candidates_idx) < nb_venues:
        best_dst, best_idx = 1e15, 0
        for ridx in range(len(right_knn['index'])):
            if ridx in candidates_idx or r.random() > 0.3:
                continue
            mask = np.array([ridx] + candidates_idx)
            weights = weighting_venues(right_knn['features'][mask, 1])
            activities = np.ones((12, 1))
            features = right_knn['features'][mask, :]
            density = features_as_density(features, weights, support)
            distance = distance_function(density, activities)
            if distance < best_dst:
                best_dst, best_idx = distance, ridx
        candidates_idx.append(best_idx)
        print('add: {}. dst = {:.4f}'.format(right_knn['index'][best_idx],
                                             best_dst))
    r_vids = [right_knn['index'][_] for _ in candidates_idx]
    return [best_dst, r_vids, [], -1], None


def get_knn_candidates(vids, left_knn, right_knn, at_least, at_most=None):
    """Return between `at_least` and `at_most` venue in right that are close (in
    the sense of euclidean distance) of the `vids` in left. Namely, it return
    their row number and their ids."""
    import heapq
    candidates = []
    candidates_id = []
    knn = right_knn['knn']
    at_most = int(at_most) or 50000
    nb_venues = min(at_most, max(len(vids)*knn, at_least))
    for idx, vid in enumerate(vids):
        _, rid, ridx, dst, _ = cn.find_closest(vid, left_knn, right_knn)
        for dst_, rid_, ridx_, idx_ in zip(dst, rid, ridx, range(knn)):
            if rid_ not in candidates_id:
                candidates_id.append(rid_)
                heapq.heappush(candidates, (dst_, idx*knn+idx_,
                                            (rid_, ridx_)))
    nb_venues = min(len(candidates), int(nb_venues))
    closest = heapq.nsmallest(nb_venues, candidates)
    mask = np.array([v[2][1] for v in closest])
    r_vids = np.array([v[2][0] for v in closest])
    return mask, r_vids


def get_neighborhood_candidates(distance_function, right_knn, metric,
                                at_most=None):
    candidates = []
    activities = np.ones((12, 1))
    weights = [1.0]
    nb_dims = right_knn['features'].shape[1]
    for idx, vid in enumerate(right_knn['index']):
        features = right_knn['features'][idx, :].reshape(1, nb_dims)
        if 'jsd' in metric:
            density = features_as_density(features, weights, RIGHT_SUPPORT)
            dst = distance_function(density, activities)
        elif 'emd' in metric:
            dst = distance_function([list(features.ravel())], weights)
        else:
            raise ValueError('unknown metric {}'.format(metric))
        candidates.append((dst, idx, vid))

    nb_venues = min(int(at_most), len(candidates))
    closest = sorted(candidates, key=lambda x: x[0])[:nb_venues]
    mask = np.array([v[1] for v in closest])
    r_vids = np.array([v[2] for v in closest])
    return mask, r_vids


def search_no_space(vids, nb_venues, distance_function, left_knn, right_knn,
                    support):
    """Find `nb_venues` in `right_knn` that are close to those in `vids` (in
    the sense of euclidean distance) and return the distance with this
    “virtual” neighborhood (for comparaison purpose)"""
    mask, r_vids = get_knn_candidates(vids, left_knn, right_knn, nb_venues)
    weights = weighting_venues(right_knn['features'][mask, 1])
    activities = np.ones((12, 1))
    features = right_knn['features'][mask, :]
    density = features_as_density(features, weights, support)
    distance = distance_function(density, activities)
    return [distance, r_vids, [], -1], None


def interpolate_distances(values_map, filename):
    """Plot the distance at every circle center and interpolate between"""
    from scipy.interpolate import griddata
    from matplotlib import pyplot as plt
    import persistent as p
    import os
    filename = os.path.join('distance_map', filename)
    x, y, z = [np.array(dim) for dim in zip(*[a for a in values_map])]
    x_ext = [x.min(), x.max()]
    y_ext = [y.min(), y.max()]
    xi = np.linspace(x_ext[0], x_ext[1], 100)
    yi = np.linspace(y_ext[0], y_ext[1], 100)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')
    fig = plt.figure(figsize=(22, 18))
    plt.contour(xi, yi, zi, 20, linewidths=0.8, colors='#282828')
    plt.contourf(xi, yi, zi, 20, cmap=plt.cm.Greens)
    plt.colorbar()
    plt.scatter(x, y, marker='o', c='#282828', s=5)
    plt.tight_layout(pad=0)
    plt.xlim(*x_ext)
    plt.ylim(*y_ext)
    plt.savefig(filename, dpi=96, transparent=False, frameon=False,
                bbox_inches='tight', pad_inches=0.01)
    p.save_var(filename.replace('.png', '.my'), values_map)
    plt.close(fig)


def batch_matching():
    """Match preselected regions of Paris into a few target cities"""
    import ujson
    global QUERY_NAME
    with open('static/cpresets.json') as infile:
        regions = ujson.load(infile)
    for metric in ['cluster']:
        print(metric)
        for neighborhood in regions.keys():
            print(neighborhood)
            rgeo = regions[neighborhood].get('geo')
            for city in ['barcelona', 'sanfrancisco']:
                print(city)
                # regions[neighborhood][city] = []
                for radius in np.linspace(200, 500, 5):
                    print(radius)
                    QUERY_NAME = '{}_{}_{}_{}.my'.format(city, neighborhood,
                                                         int(radius),
                                                         metric)
                    res, values, _ = best_match('paris', city, rgeo, radius,
                                                metric=metric).next()
                    continue
                    distance, r_vids, center, radius = res
                    print(distance)
                    if center is None:
                        result = {'dst': distance, 'metric': metric,
                                  'nb_venues': 0}
                    else:
                        center = cities.euclidean_to_geo(city, center)
                        result = {'geo': {'type': 'circle',
                                          'center': center, 'radius': radius},
                                  'dst': distance, 'metric': metric,
                                  'nb_venues': len(r_vids)}
                    regions[neighborhood][city].append(result)
                    # outname = '{}_{}_{}_{}.png'.format(city, neighborhood,
                    #                                    int(radius), metric)
                    # interpolate_distances(values, outname)
                with open('static/cpresets.js', 'w') as out:
                    out.write('var PRESETS =' + ujson.dumps(regions) + ';')


def find_promising_seeds(good_ids, venues_infos, method, right):
    """Try to find high concentration of `good_ids` venues among all
    `venues_infos` using one of the following methods:
    ['dbscan'|'discrepancy'].
    Return a list of convex hulls with associated list of good and bad
    venues id"""
    vids, _, venues_loc = venues_infos.all()
    significant_id = {vid: loc for vid, loc in i.izip(vids, venues_loc)
                      if vid in right['index']}
    good_loc = np.array([significant_id[v] for v in good_ids])
    bad_ids = [v for v in significant_id.iterkeys() if v not in good_ids]
    bad_loc = np.array([significant_id[v] for v in bad_ids])
    if method == 'discrepancy':
        hulls, gcluster, bcluster = discrepancy_seeds((good_ids, good_loc),
                                                      (bad_ids, bad_loc),
                                                      np.array(venues_loc))
    elif method == 'dbscan':
        hulls, gcluster, bcluster = dbscan_seeds((good_ids, good_loc),
                                                 (bad_ids, bad_loc))
    else:
        raise ValueError('{} is not supported'.format(method))
    clusters = zip(hulls, gcluster, bcluster)
    return sorted(clusters, key=lambda x: len(x[1]), reverse=True)


def discrepancy_seeds(goods, bads, all_locs):
    """Find regions with concentration of good points compared with bad
    ones."""
    import spatial_scan as sps
    size = 50
    support = 8
    sps.GRID_SIZE = size
    sps.TOP_K = 500

    xedges, yedges = [np.linspace(low, high, size+1)
                      for low, high in zip(np.min(all_locs, 0),
                                           np.max(all_locs, 0))]
    bins = (xedges, yedges)
    good_ids, good_loc = goods
    bad_ids, bad_loc = bads
    count, _, _ = np.histogram2d(good_loc[:, 0], good_loc[:, 1], bins=bins)
    measured = count.T.ravel()
    count, _, _ = np.histogram2d(bad_loc[:, 0], bad_loc[:, 1], bins=bins)
    background = count.T.ravel()
    total_b = np.sum(background)
    total_m = np.sum(measured)
    discrepancy = sps.get_discrepancy_function(total_m, total_b, support)

    def euc_index_to_rect(idx):
        """Return the bounding box of a grid's cell defined by its
        `idx`"""
        i = idx % size
        j = idx / size
        return [xedges[i], yedges[j], xedges[i+1], yedges[j+1]]
    sps.index_to_rect = euc_index_to_rect

    top_loc = sps.exact_grid(np.reshape(measured, (size, size)),
                             np.reshape(background, (size, size)),
                             discrepancy, sps.TOP_K,
                             sps.GRID_SIZE/8)
    merged = sps.merge_regions(top_loc)

    gcluster = []
    bcluster = []
    hulls = []
    for region in merged:
        gcluster.append([id_ for id_, loc in zip(good_ids, good_loc)
                         if region[1].contains(sgeo.Point(loc))])
        bcluster.append([id_ for id_, loc in zip(bad_ids, bad_loc)
                         if region[1].contains(sgeo.Point(loc))])
        hulls.append(region[1].convex_hull)
    return hulls, gcluster, bcluster


def dbscan_seeds(goods, bads):
    """Find regions with concentration of good points."""
    from scipy.spatial import ConvexHull
    import sklearn.cluster as cl
    good_ids, good_loc = goods
    bad_ids, bad_loc = bads
    labels = cl.DBSCAN(eps=150, min_samples=8).fit_predict(good_loc)
    gcluster = []
    bcluster = []
    hulls = []
    for cluster in range(len(np.unique(labels))-1):
        points = good_loc[labels == cluster, :]
        hull = sgeo.Polygon(points[ConvexHull(points).vertices])
        gcluster.append(list(i.compress(good_ids, labels == cluster)))
        bcluster.append([id_ for id_, loc in zip(bad_ids, bad_loc)
                         if hull.contains(sgeo.Point(loc))])
        hulls.append(hull)
    return hulls, gcluster, bcluster

if __name__ == '__main__':
    # pylint: disable=C0103
    batch_matching()
    import sys
    sys.exit()
    import arguments
    args = arguments.two_cities().parse_args()
    origin, dest = args.origin, args.dest
    user_input = {"type": "Polygon",
                  "coordinates": [[[2.3006272315979004, 48.86419005209702],
                                   [2.311570644378662, 48.86941264251879],
                                   [2.2995758056640625, 48.872983451383305],
                                   [2.3006272315979004, 48.86419005209702]]]}
    get_seed_regions(origin, dest, user_input)
    sys.exit()
    res, values, _ = best_match(origin, dest, user_input, 800,
                                metric='jsd-nospace').next()
    distance, r_vids, center, radius = res
    print(distance)
    for _ in sorted(r_vids):
        print("'{}',".format(str(_)))
    # print(distance, cities.euclidean_to_geo(dest, center))
    # interpolate_distances(values, origin+dest+'.png')

    # KDE preprocessing
    # given all tweets, bin them according to time.
    # Then run KDE on each bin, and compute a normalized grid in both cities
    # (it's not cheap, but it's amortized over all queries)
    # Then, when given a query, compute its average value for each time
    # set a narrow range around each value and take the intersection of all
    # point within this range in the other city.
    # Increase range until we get big enough surface
    # (or at least starting point)

    # Better quality
    # best_match()
    # gather query and cities info
    # OPTIONNAL: reweight query venues to remove outliers
    # get individual candidate (at_least, at_most, distance ∈ emd, jsd, knn)
    # find seed among candidate (DBSCAN, K-means, discrepancy)
    # grow seeds into region (in a greedy manner)
    # return polygonal geo hull

    # Metric Learning
    # I have six points in Paris, find close and distant ones in San Francisco
    # and Barcelona to tune theta in JSD and EMD.
