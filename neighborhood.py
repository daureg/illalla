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
import shapely.geometry as geo
# pylint: disable=E1101
# pylint: disable=W0621


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
    projected = geo.Polygon(cities.GEO_TO_2D[city](coords))
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


@profile
def features_as_density(features, weights, support, bins=10):
    """Turn raw `features` into probability distribution over each dimension,
    with respect to `weights`."""
    @profile
    def get_bins(dim):
        extent = support[dim][1] - support[dim][0]
        size = 1.0/bins
        return [support[dim][0] + j*size*extent for j in range(bins+1)]

    return np.vstack([np.histogram(features[:, i], weights=weights,
                                   bins=get_bins(i))[0]
                      for i in range(features.shape[1])])


def features_as_lists(features):
    """Turn numpy `features` into a list of list, suitable for emd
    function."""
    return [list(row) for row in features]


@profile
def weighting_venues(values):
    """Transform `values` into a list of positive weights that sum up to 1."""
    from sklearn.preprocessing import MinMaxScaler
    scale = MinMaxScaler()
    size = values.size
    scaled = scale.fit_transform(np.power(values, .2).reshape((size, 1)))
    return scaled.ravel()/np.sum(scaled)


@profile
def gather_entities(surrounding, center, radius, belongs_to, threshold=0):
    """Filter points in `surrounding` that belong to the given region."""
    ids, info, locs = surrounding.around(center, radius)
    info = len(ids)*[0, ] if len(info) == 0 else list(info[0])
    if len(ids) < threshold:
        return None, None
    if belongs_to is None:
        return ids, info
    is_inside = lambda t: belongs_to(geo.Point(t[2]))
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
USE_EMD, CITY_SUPPORT, DISTANCE_FUNCTION, RADIUS = None, None, None, None


@profile
def one_cell(args):
    cx, cy, idx, idy = args
    center = [cx, cy]
    contains = None
    candidate = describe_region(center, RADIUS, contains,
                                SURROUNDINGS, CITY_FEATURES,
                                THRESHOLD)
    features, c_times, weights, c_vids = candidate
    if features is not None:
        if USE_EMD:
            c_density = features_as_lists(features)
        else:
            c_density = features_as_density(features, weights,
                                            CITY_SUPPORT)
        distance = DISTANCE_FUNCTION(c_density,
                                     weights if USE_EMD else c_times)
        return [cx, cy, distance, c_vids]
    else:
        return [None, None, None, None]


@profile
def brute_search(city_desc, hsize, distance_function, threshold,
                 metric='jsd'):
    """Move a sliding circle over the whole city and keep track of the best
    result."""
    global SURROUNDINGS, CITY_FEATURES, THRESHOLD, RADIUS
    global USE_EMD, CITY_SUPPORT, DISTANCE_FUNCTION
    import multiprocessing
    RADIUS = hsize
    THRESHOLD = threshold
    USE_EMD = metric == 'emd'
    city_size, CITY_SUPPORT, CITY_FEATURES, city_infos = city_desc
    SURROUNDINGS, bounds = city_infos
    DISTANCE_FUNCTION = distance_function
    minx, miny, maxx, maxy = bounds
    nb_x_step = int(3*np.floor(city_size[0]) / hsize + 1)
    nb_y_step = int(3*np.floor(city_size[1]) / hsize + 1)
    best = [1e20, [], [], RADIUS]
    res_map = []
    pool = multiprocessing.Pool(4)

    x_steps = np.linspace(minx+hsize, maxx-hsize, nb_x_step)
    y_steps = np.linspace(miny+hsize, maxy-hsize, nb_y_step)
    x_vals, y_vals = np.meshgrid(x_steps, y_steps)
    to_cell_arg = lambda _: (float(_[1][0]), float(_[1][1]), _[0] % nb_x_step,
                             _[0]/nb_y_step)
    cells = i.imap(to_cell_arg, enumerate(i.izip(np.nditer(x_vals),
                                                 np.nditer(y_vals))))
    res = pool.map(one_cell, cells)
    res_map = []
    for cell in res:
        if cell[0] is None:
            continue
        res_map.append(cell[:3])
        if cell[2] < best[0]:
            best = [cell[2], cell[3], [cell[0], cell[1]], RADIUS]
    yield best, res_map, 1.0


def best_match(from_city, to_city, region, tradius, progressive=False,
               metric='jsd'):
    """Try to match a `region` from `from_city` to `to_city`. If progressive,
    yield intermediate result."""
    assert metric in ['jsd', 'emd', 'jsd-nospace', 'jsd-greedy']
    from emd import emd
    from emd_dst import dist_for_emd
    # Load info of the first city
    left = cn.gather_info(from_city, knn=1)
    left_infos = load_surroundings(from_city)
    minx, miny, maxx, maxy = left_infos[1]
    # left_city_size = (maxx - minx, maxy - miny)
    left_support = features_support(left['features'])

    # Compute info about the query region
    center, radius, bounds, contains = polygon_to_local(from_city, region)
    query = describe_region(center, radius, contains, left_infos[0], left)
    features, times, weights, vids = query
    print('{} venues in query region.'.format(len(vids)))
    yield len(vids), None, None
    raise Exception('done!')
    if 'emd' in metric:
        query_num = features_as_lists(features)
    else:
        query_num = features_as_density(features, weights, left_support)
    venue_proportion = 1.0*len(vids) / left['features'].shape[0]
    minx, miny, maxx, maxy = bounds
    # average_dim = ((maxx - minx)/left_city_size[0] +
    #                (maxy - miny)/left_city_size[1])*0.5

    # And use them to define the metric that will be used
    theta = np.ones((1, left['features'].shape[1]))

    if 'emd' in metric:
        @profile
        def regions_distance(r_features, r_weigths):
            return emd((query_num, map(float, weights)),
                       (r_features, map(float, r_weigths)),
                       lambda a, b: float(dist_for_emd(a, b)))
    else:
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

    # given extents, compute treshold and size of candidate
    threshold = 0.7 * venue_proportion * right['features'].shape[0]
    hsize = tradius  # TODO function of average_dim and right_city_size
    right_desc = [right_city_size, right_support, right, right_infos]

    res, vals = None, None
    if metric.endswith('-nospace'):
        res, vals = search_no_space(vids, 10.0/7*threshold, regions_distance,
                                    left, right, right_support)
    elif metric.endswith('-greedy'):
        res, vals = greedy_search(10.0/7*threshold, regions_distance, right,
                                  right_support)
    else:
        # Use case for https://docs.python.org/3/whatsnew/3.3.html#pep-380
        for res, vals, progress in brute_search(right_desc, hsize,
                                                regions_distance, threshold,
                                                metric=metric):
            if progressive:
                yield res, vals, progress
            else:
                print(progress, end='\t')
    yield res, vals, 1.0


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


def search_no_space(vids, nb_venues, distance_function, left_knn, right_knn,
                    support):
    """Find `nb_venues` in `right_knn` that are close to those in `vids` (in
    the sense of euclidean distance) and return the distance with this
    “virtual” neighborhood (for comparaison purpose)"""
    import heapq
    candidates = []
    candidates_id = []
    knn = right_knn['knn']
    nb_venues = min(len(vids)*knn, int(nb_venues)+30)
    for idx, vid in enumerate(vids):
        _, rid, ridx, dst, _ = cn.find_closest(vid, left_knn, right_knn)
        for dst_, rid_, ridx_, idx_ in zip(dst, rid, ridx, range(knn)):
            if rid_ not in candidates_id:
                candidates_id.append(rid_)
                heapq.heappush(candidates, (dst_, idx*knn+idx_,
                                            (rid_, ridx_)))
    closest = heapq.nsmallest(nb_venues, candidates)
    mask = np.array([v[2][1] for v in closest])
    r_vids = np.array([v[2][0] for v in closest])
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
    """Match preselected regions of Paris into Helsinki and Barcelona"""
    import ujson
    with open('static/presets.json') as infile:
        regions = ujson.load(infile)
    for neighborhood in regions.iterkeys():
        print(neighborhood)
        rgeo = regions[neighborhood].get('geo')
        for city in ['helsinki', 'barcelona']:
            print(city)
            regions[neighborhood][city] = []
            for metric in ['jsd', 'emd']:
                print(metric)
                for radius in np.linspace(350, 1100, 6):
                    print(radius)
                    res, values, _ = best_match('paris', city, rgeo, radius,
                                                metric=metric).next()
                    distance, r_vids, center, radius = res
                    print(distance)
                    center = cities.euclidean_to_geo(city, center)
                    result = {'geo': {'type': 'circle',
                                      'center': center, 'radius': radius},
                              'dst': distance, 'metric': metric,
                              'nb_venues': len(r_vids)}
                    regions[neighborhood][city].append(result)
                    outname = '{}_{}_{}_{}.png'.format(city, neighborhood,
                                                       int(radius), metric)
                    interpolate_distances(values, outname)
    with open('static/fapresets.js', 'w') as out:
        out.write('var PRESETS =' + ujson.dumps(regions) + ';')

if __name__ == '__main__':
    # pylint: disable=C0103
    # batch_matching()
    # import sys
    # sys.exit()
    import arguments
    args = arguments.two_cities().parse_args()
    origin, dest = args.origin, args.dest
    user_input = {"type": "Polygon",
                  "coordinates": [[[2.3006272315979004, 48.86419005209702],
                                   [2.311570644378662, 48.86941264251879],
                                   [2.2995758056640625, 48.872983451383305],
                                   [2.3006272315979004, 48.86419005209702]]]}
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
    # Increase range until we get big enough surface (or at least starting point)

    # Better quality
    # best_match()
    # gather query and cities info
    # reweight query venues to remove outliers
    # get individual candidate (at_least, at_most, distance ∈ emd, jsd, knn)
    # find seed among candidate (DBSCAN, K-means, discrepancy)
    # grow seeds into region (in a greedy manner)
    # return polygonal geo hull

    # Metric Learning
    # I have six points in Paris, find close and distant ones in San Francisco
    # and Barcelona to tune theta in JSD and EMD.
