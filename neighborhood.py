#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Match polygonal regions between cities
Input:
    name of two cities
    a list of coordinates that make up a polygon in one city
Output:
    a list of coordinates that make up a polygon in the other city
"""
import cities
import ClosestNeighbor as cn
import explore as xp
import numpy as np
import utils as u
from shapely.geometry import Polygon


def load_surroundings(city):
    """Load projected coordinates and extra field of all venues, checkins and
    photos within `city`."""
    import persistent as p
    surroundings =  [p.load_var('{}_l{}s.my'.format(city, kind))
                     for kind in ['venue', 'checkin', 'photo']]
    venues_pos = np.vstack(surroundings[0].values())
    city_extent = list(np.min(venues_pos, 0)) + list(np.max(venues_pos, 0))
    return surroundings, city_extent


def polygon_to_local(city, geojson):
    """Convert a `geojson` geometry to a local bounding box in `city`, and
    return its center, radius and a predicate indicating membership."""
    assert geojson['type'] == 'Polygon'
    projected = Polygon(cities.GEO_TO_2D[city](geojson['coordinates']))
    minx, miny, maxx, maxy = projected.bounds
    center = projected.centroid
    radius = max(maxx - minx, maxy - miny)*0.5
    average = (maxx - minx + maxy - miny)*0.5
    return center, radius, projected.bounds, projected.contains


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
    cids, ctime = gather_entities(scheckins, center, radius, belongs_to,
                                  threshold)
    pids, ptime = gather_entities(sphotos, center, radius, belongs_to,
                                  threshold)
    mask = np.where(np.in1d(city_fv['index'], vids))[0]
    weights = weighting_venues(city_fv['features'][mask, 1])
    time_activity = lambda visits: xp.aggregate_visits(visits, 1, 4)[0]
    activities = np.hstack(xp.to_frequency(time_activity(ctime)),
                           xp.to_frequency(time_activity(ptime)))
    # TODO compute it only once
    extents = zip(np.min(city_fv['features'], 0),
                  np.max(city_fv['features'], 0))
    return city_fv['features'][mask, :], activities, weights, vids, extents


def features_as_density(features, weights, support, bins=10):
    """Turn raw `features` into probability distribution over each dimension,
    with respect to `weights`."""
    def get_bins(dim):
        extent = support[dim][1] - support[dim][0]
        return [support[dim][0] + (1.0*j/bins)*extent for j in range(bins+1)]

    return np.vstack([np.histogram(features[:, i], weights=weights,
                                   bins=get_bins(i))[0]
                     for i in range(features.shape[1])])


def weighting_venues(values):
    """Transform `values` into a list of positive weights that sum up to 1."""
    from sklearn.preprocessing import MinMaxScaler
    scale = MinMaxScaler()
    size = values.size
    scaled = scale.fit_transform(np.power(values, .2).reshape((size, 1)))
    return scaled.ravel()/np.sum(scaled)


def gather_entities(surrounding, center, radius, belongs_to, threshold):
    """Filter points in `surrounding` that belong to the given region."""
    ids, info, locs = surrounding.around(center, radius)
    info = info[0]
    if len(ids) < threshold:
        return None, None
    ids[:], info[:], locs[:] = zip(*(i.ifilter(lambda t: belongs_to(t[2]),
                                               i.izip(ids, info, locs))))
    if len(ids) < threshold:
        return None, None
    return ids, info


def jensen_shannon_divergence(P, Q):
    """Compute JSD(P || Q) as defined in
    https://en.wikipedia.org/wiki/Jensen–Shannon_divergence """
    avg = 0.5*(P + Q)
    avg_H = 0.5*(u.compute_entropy(P) + u.compute_entropy(Q))
    return u.compute_entropy(avg) - avg_H


def proba_distance(density1, global1, density2, global2, theta):
    """Compute total distances between all distributions"""
    proba = np.dot(theta, [jensen_shannon_divergence(p, q)
                           for p, q in zip(density1, density2)])
    return proba + np.linalg.norm(global1 - global2)


if __name__ == '__main__':
    # pylint: disable=C0103
    import arguments
    args = arguments.two_cities().parse_args()
    origin, dest = args.origin, args.dest
    user_input = {
        "type": "Polygon",
        "coordinates": [[
            [2.35015, 48.85893],
            [2.35615, 48.85893],
            [2.35615, 48.85293],
            [2.35015, 48.85293],
            [2.35015, 48.85893]
        ]]
    }
    left = cn.gather_info(args.origin, knn=1)
    right = cn.gather_info(args.dest, knn=1)
    left_infos = load_surroundings(origin)
    right_infos = load_surroundings(dest)
    center, radius, bounds, contains = polygon_to_local(origin, user_input)
    query = describe_region(center, radius, contains, left_infos[0], left)
    return city_fv['features'][mask, :], activities, weights, vids, extents
    features, times, weights, vids, extents = query
    query_num = features_as_density(features, weights, extents)

    def regions_distance(r_density, r_global):
        """Return distance of a region from `query_num`."""
        return proba_distance(query_num, times, r_density, r_global,
                              np.ones((1, query_num.shape[1])))
    minx, miny, maxx, maxy = left_infos[1]
    venue_proportion = 1.0*len(vids) / left['features'].shape[0]
    # given extents, compute treshold and size of candidate
    # then move candidate, describe, compute distance and keep track of best

    # get coordinates of venues in best candidate and return corresponding
    # bounding box as GeoJSON.
    db = cm.connect_to_db('foursquare', args.host, args.port)[0]
