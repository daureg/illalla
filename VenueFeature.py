#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Try to describe venue by various features."""
import prettyplotlib as ppl
import matplotlib.pyplot as plt
from collections import Counter, defaultdict, OrderedDict
from sklearn.neighbors import KernelDensity
import CommonMongo as cm
import FSCategories as fsc
import explore as xp
import numpy as np
import pandas as pd
import utils as u
import random as r
import itertools
import scipy.cluster.vq as cluster
from scipy.stats import multivariate_normal
import re
import string
NOISE = re.compile(r'[\s'+string.punctuation+r']')
DB = None
CLIENT = None
LEGEND = 'v^<>s*xo|8d+'
CATS = ['Arts & Entertainment', 'College & University', 'Food',
        'Nightlife Spot', 'Outdoors & Recreation', 'Shop & Service',
        'Professional & Other Places', 'Residence', 'Travel & Transport']


def venues_info(vids, visits=None, visitors=None, density=None, depth=10):
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
    res['Den'] = density(points)
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
    max_density = approximate_maximum_density(kde, surround.venues[:, :2])
    # pylint: disable=E1101
    return lambda xy: np.exp(kde.score_samples(xy))/max_density


def approximate_maximum_density(kde, venues, precision=128):
    """Evaluate the kernel on a grid and return the max value."""
    # pylint: disable=E1101
    xgrid = np.linspace(np.min(venues[:, 0]), np.max(venues[:, 0]), precision)
    ygrid = np.linspace(np.min(venues[:, 1]), np.max(venues[:, 1]), precision)
    X, Y = np.meshgrid(xgrid, ygrid)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    estim = np.exp(kde.score_samples(xy))
    return estim.max()


def smoothed_location(loc, center, radius, city):
    """Return a list of weight (obtained by a 2D Gaussian with `radius`)
    corresponding to the relative distance of points in `loc` with
    `center`."""
    project = cm.cities.GEO_TO_2D[city]
    center = center if 'coordinates' not in center else center['coordinates']
    center = project(reversed(center))[:2]
    ploc = project(np.array(loc)) - np.tile(center, (len(loc), 1))
    smooth = multivariate_normal([0, 0], (radius/2.5)*np.eye(2))
    return smooth.pdf(ploc/20)/smooth.pdf([0, 0])


def full_surrounding(vid, city=None, radius=350):
    """Return a list of photos, checkins and venues categories in a `radius`
    around `vid`, within `city`."""
    self = DB.venue.find_one({'_id': vid}, {'loc': 1, 'city': 1})
    city, position = city or self['city'], self['loc']
    ball = {'$geometry': position, '$maxDistance': radius}
    return categories_repartition(city, ball)
    photos = CLIENT.world.photos({'hint': city, 'loc': {'$near': ball}},
                                 {'venue': 1, 'taken': 1, 'loc': 1})
    pvenue, ptime, ploc = zip(*[(p['venue'], p['taken'],
                                 list(reversed(p['loc']['coordinates'])))
                                for p in photos])
    checkins = DB.checkin.find({'city': city, 'loc': {'$near': ball}},
                               {'time': 1, 'loc': 1})
    ctime, cloc = zip(*[(c['time'], list(reversed(c['loc']['coordinates'])))
                        for c in checkins])


def categories_repartition(city, ball=None):
    """Return the distribution of top level Foursquare categories in
    `ball` (or the whole `city` if None)."""
    if ball:
        position, radius = ball['$geometry'], ball['$maxDistance']
        geo = {'$near': ball}
    else:
        geo = {'$ne': None}
    neighbors = DB.venue.find({'city': city, 'loc': geo},
                              {'cat': 1, 'cats': 1, 'loc': 1})
    vcats, vloc = zip(*[([v['cat']] + v['cats'],
                         list(reversed(v['loc']['coordinates'])))
                        for v in neighbors])
    smoothed_loc = itertools.cycle([1.0])
    if ball:
        smoothed_loc = smoothed_location(vloc, position, radius, city)
    distrib = defaultdict(int)
    for own_cat, weight in zip(vcats, smoothed_loc):
        top_cat = [parenting_cat(c) for c in own_cat if c]
        for cat in top_cat:
            distrib[cat] += weight
    distrib = np.array([distrib[c] for c in CATS])
    # Can't be zero because there is always at least the venue itself in
    # surrounding.
    # TODO: maybe it would be more informative to return how it deviate from
    # the global distribution.
    return distrib / np.sum(distrib)


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


def named_ticks(kind, offset=0, chunk=3):
    """Return ticks label for kind in ('day', 'week', 'mix')."""
    if kind is 'day':
        period = lambda i: '{}--{}'.format(i % 24, (i+chunk) % 24)
        return [period(i) for i in range(0+offset, 24+offset, chunk)]
    days = 'mon tue wed thu fri sat sun'.split()
    if kind is 'week':
        return days
    if kind is 'mix':
        period = '1 2 3'.split()
        return [d+''+p for d in days for p in period]
    raise ValueError('`kind` arguments is not valid')


def draw_classes(centroid, offset, chunk=3):
    """Plot each time patterns in `centroid`."""
    size = centroid.shape[0]
    for i, marker in zip(range(size), LEGEND[:size]):
        ppl.plot(centroid[i, :], marker+'-', ms=9)
    if centroid.shape[1] == 24/chunk:
        plt.xticks(range(24/chunk), named_ticks('day', offset, chunk))
    else:
        plt.xticks(range(7*3), named_ticks('mix'))


def get_distorsion(ak, kl, sval):
    """Compute the sum of euclidean distance from `sval` to its
    centroid"""
    return np.sum(np.linalg.norm(ak[kl, :] - sval, axis=1))

if __name__ == '__main__':
    # pylint: disable=C0103
    import arguments
    args = arguments.city_parser().parse_args()
    city = args.city
    DB, CLIENT = cm.connect_to_db('foursquare', args.host, args.port)

    # pylint: disable=E1101
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
