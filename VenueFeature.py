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
    """Return the name of category id `cat` (or name), stopping at level
    `depth`."""
    if not cat:
        return None
    _, path = fsc.search_categories(cat)
    cat_is_name = fsc.choose_type(cat)
    answer = path[depth] if len(path) > depth else path[-1]
    if cat_is_name:
        return answer
    return fsc.CAT_TO_ID[:answer]


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
    return len(photos), kind, classes, np.argmin(classes)


def named_ticks(kind, offset=0):
    """Return ticks label for kind in ('day', 'week', 'mix')."""
    if kind is 'day':
        period = lambda i: '{}--{}'.format(i % 24, (i+3) % 24)
        return [period(i) for i in range(0+offset, 24+offset, 3)]
    days = 'mon tue wed thu fri sat sun'.split()
    if kind is 'week':
        return days
    if kind is 'mix':
        period = '1 2 3'.split()
        return [d+''+p for d in days for p in period]
    raise ValueError('`kind` arguments is not valid')


def draw_classes(centroid, offset):
    """Plot each time patterns in `centroid`."""
    size = centroid.shape[0]
    for i, marker in zip(range(size), legend[:size]):
        pp.plot(centroid[i, :], marker+'-', ms=11)
    if centroid.shape[1] == 8:
        pp.xticks(range(8), named_ticks('day', offset))
    else:
        pp.xticks(range(7*3), named_ticks('mix'))


if __name__ == '__main__':
    #pylint: disable=C0103
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
