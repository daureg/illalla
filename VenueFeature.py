#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Try to describe venue by various features."""
from matplotlib import pyplot as pp
import CommonMongo as cm
import FSCategories as fsc
import explore as xp
import numpy as np
import pandas as pd
import utils as u
import random as r
import scipy.cluster.vq as cluster
DB = None


def parenting_cat(cat, depth=1):
    """Return the name of category id `cat`, stopping at level `depth`."""
    if not cat:
        return None
    _, path = fsc.search_categories(cat)
    if len(path) > depth:
        return fsc.CAT_TO_ID[:path[depth]]
    return fsc.CAT_TO_ID[:path[-1]]


def get_loc(vid):
    """Return coordinated of the venue `vid` (or None if it's not in DB)."""
    res = DB.venue.find_one({'_id': vid}, {'loc': 1})
    if res:
        return u.get_nested(res, ['loc', 'coordinates'])
    return None


def get_venue(vid, depth=1):
    """Return a textual description of venue `vid` or None."""
    venue = DB.venue.find_one({'_id': vid}, {'cat': 1, 'name': 1})
    if not venue:
        return None
    cat = parenting_cat(venue.get('cat'), depth)
    venue['cat'] = cat or '???'
    return (venue['cat'], venue['name'], vid)


def photos_around(id_, centroid, offset, daily, radius=200):
    """Gather photos timestamp in a `radius` around `id_` and return its time
    pattern (`daily` or not), and its distance to every `centroid`."""
    center = get_loc(id_)
    photos = xp.get_visits(client, xp.Entity.photo, ball=(center, radius))
    kind = xp.to_frequency(xp.aggregate_visits(photos.values(), offset)[daily])
    nb_class = centroid.shape[0]
    classes = np.linalg.norm(np.tile(kind, (nb_class, 1)) - centroid, axis=1)
    return kind, classes, np.argmin(classes)


def draw_classes(centroid, offset):
    """Plot each time patterns in `centroid`."""
    size = centroid.shape[0]
    for i, marker in zip(range(size), legend[:size]):
        pp.plot(centroid[i, :], marker+'-', ms=11)
    if centroid.shape[1] == 8:
        period = lambda i: '{}--{}'.format(i % 24, (i+3) % 24)
        pp.xticks(range(8), [period(i)
                             for i in range(0+offset, 24+offset, 3)])
    else:
        days = 'mon tue wed thu fri sat sun'.split()
        period = '1 2 3'.split()
        pp.xticks(range(7*3), [d+''+p for d in days for p in period])


if __name__ == '__main__':
    import arguments
    args = arguments.city_parser().parse_args()
    city = args.city
    DB, client = cm.connect_to_db('foursquare', args.host, args.port)

    legend = 'v^<>s*xo|8d+'

    def get_distorsion(ak, kl, sval):
        return np.sum(np.linalg.norm(ak[kl, :] - sval, axis=1))

    do_cluster = lambda val, k: cluster.kmeans2(val, k, 20, minit='points')

    def getclass(c, kl, visits):
        return {v[0]: v[1] for v, k in zip(visits.iteritems(), kl) if k == c}

    def peek_at_class(c, kl, visits):
        sample = r.sample([get_venue(i)
                           for i in getclass(c, kl, visits).keys()], 15)
        return pd.DataFrame({'cat': [_[0] for _ in sample],
                             'name': [_[1] for _ in sample],
                             'id': [_[2] for _ in sample]})
    # TODO; check if results are better when withening data beforehand (but
    # then we need the standard deviation of each feature to classify new
    # observations)
    # disto = [comp_disto(*cluster.kmeans2(sval, k, 25, minit='points'))
    #          for k in range(2, 15)]
    # K = 6
    # ak, kl = do_cluster(sval, K)
    # print(np.sort(np.bincount(kl)))
    # print(list(enumerate(legend[:K])))
    # db = DB
    # shift = 1
    # daily = 1
    # venue_visits = xp.get_visits(client, xp.Entity.venue, city)
    # sig = {k: xp.to_frequency(xp.aggregate_visits(v, shift)[daily])
    #        for k, v in venue_visits.iteritems() if len(v) > 5}
    # sval = np.array(sig.values())
    # print(np.mean(sval, 0))
