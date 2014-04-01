#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Try to describe venue by various features."""
from matplotlib import pyplot as pp
from collections import Counter, defaultdict, OrderedDict
from sklearn.neighbors import KernelDensity
import CommonMongo as cm
import FSCategories as fsc
import explore as xp
import numpy as np
import pandas as pd
import utils as u
import random as r
import scipy.cluster.vq as cluster
import re
import string
NOISE = re.compile(r'[\s'+string.punctuation+r']')
DB = None
CLIENT = None


def venues_info(vids, density=None, visits=None, visitors=None, depth=10):
    """Return various info about from the venue ids `vids`."""
    tags = defaultdict(int)
    city = DB.venue.find_one({'_id': vids[0]})['city']
    visits = visits or xp.get_visits(CLIENT, xp.Entity.venue, city)
    visitors = visitors or xp.get_visitors(CLIENT, city)
    density = density or estimate_density(city)
    venues = list(DB.venue.find({'_id': {'$in': vids}},
                                {'cat': 1, 'name': 1, 'loc': 1,
                                 'price': 1, 'rating': 1, 'tags': 1,
                                 'likes': 1, 'usersCount': 1,
                                 'checkinsCount': 1}))

    res = pd.DataFrame(index=[_['_id'] for _ in venues])

    def add_col(field):
        res[field.replace('Count', '')] = [_[field] for _ in venues]
    for field in ['name', 'price', 'rating', 'likes',
                  'usersCount', 'checkinsCount']:
        add_col(field)
    res['tags'] = [[normalized_tag(t) for t in _['tags']] for _ in venues]
    loc = [_['loc']['coordinates'] for _ in venues]
    # res['loc'] = loc
    res['cat'] = [parenting_cat(_['cat'], depth) for _ in venues]
    res['vis'] = [len(visits[id_]) for id_ in res.index]
    res['H'] = [venue_entropy(visitors[id_]) for id_ in res.index]
    coords = np.fliplr(np.array(loc))
    points = cm.cities.GEO_TO_2D[city](coords)
    res['Den'] = density(points)/1e-6
    for venue in venues:
        for tag in venue['tags']:
            tags[normalized_tag(tag)] += 1
    return res, OrderedDict(sorted(tags.iteritems(), key=lambda x: x[1],
                                   reverse=True))


def estimate_density(city):
    """Return a Gaussian KDE of venues in `city`."""
    kde = KernelDensity(bandwidth=175, rtol=1e-4)
    surround = xp.build_surrounding(DB.venue, city, likes=1, checkins=5)
    kde.fit(surround.venues[:, :2])
    return lambda xy: np.exp(kde.score_samples(xy))


def venue_entropy(visitors):
    """Compute the entropy of venue given the list of its `visitors`."""
    # pylint: disable=E1101
    c = np.array(Counter(visitors).values(), dtype=float)
    N = np.sum(c)
    return np.log(N) - np.sum(c*np.log(c))/N


def normalized_tag(tag):
    """normalize `tag` by removing punctuation and space character."""
    return NOISE.sub('', tag).lower()


def count_tags(tags):
    """Count occurence of a list of list of tags."""
    return Counter([normalized_tag(t) for oneset in tags for t in oneset])


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
    photos = xp.get_visits(CLIENT, xp.Entity.photo, ball=(center, radius))
    kind = xp.to_frequency(xp.aggregate_visits(photos.values(), offset)[daily])
    nb_class = centroid.shape[0]
    # pylint: disable=E1101
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
    DB, CLIENT = cm.connect_to_db('foursquare', args.host, args.port)

    legend = 'v^<>s*xo|8d+'

    # pylint: disable=E1101
    def get_distorsion(ak, kl, sval):
        """Compute the sum of euclidean distance from `sval` to its
        centroid"""
        return np.sum(np.linalg.norm(ak[kl, :] - sval, axis=1))

    do_cluster = lambda val, k: cluster.kmeans2(val, k, 20, minit='points')

    def getclass(c, kl, visits):
        """Return {id: time pattern} of the venues in classs `c` of
        `kl`."""
        return {v[0]: v[1] for v, k in zip(visits.iteritems(), kl) if k == c}

    def peek_at_class(c, kl, visits, k=15):
        """Return a table of `k` randomly chosen venues in class `c` of
        `kl`."""
        sample = r.sample([get_venue(i)
                           for i in getclass(c, kl, visits).keys()], k)
        return pd.DataFrame({'cat': [_[0] for _ in sample],
                             'name': [_[1] for _ in sample],
                             'id': [_[2] for _ in sample]})
