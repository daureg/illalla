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
import scipy.stats as stats
import scipy.io as sio
import scipy.cluster.vq as cluster
try:
    from scipy.stats import multivariate_normal
except ImportError:
    from _multivariate import multivariate_normal
import re
import string
from ProgressBar import AnimatedProgressBar
NOISE = re.compile(r'[\s'+string.punctuation+r']')
DB = None
CLIENT = None
LEGEND = 'v^<>s*xo|8d+'
CATS = ['Arts & Entertainment', 'College & University', 'Food',
        'Nightlife Spot', 'Outdoors & Recreation', 'Shop & Service',
        'Professional & Other Places', 'Residence', 'Travel & Transport']


def geo_project(city, entities):
    """Return {id: euclidean projection in `city`} for objects in
    `entities`."""
    ids, loc = zip(*[(_['_id'], list(reversed(_['loc']['coordinates'])))
                     for _ in entities])
    project = cm.cities.GEO_TO_2D[city]
    return dict(zip(ids, project(np.array(loc))))


@u.memodict
def is_event(cat_id):
    """Does `cat_id` represent an event."""
    return cat_id in fsc.get_subcategories('Event', fsc.Field.id)


def describe_city(city):
    """Compute feature vector for selected venue in `city`."""
    print("Gather information about {}.".format(city))
    lvenues = geo_project(city, DB.venue.find({'city': city}, {'loc': 1}))
    lcheckins = geo_project(city, DB.checkin.find({'city': city}, {'loc': 1}))
    lphotos = geo_project(city, CLIENT.world.photos.find({'hint': city},
                                                         {'loc': 1}))
    visits = xp.get_visits(CLIENT, xp.Entity.venue, city)
    visitors = xp.get_visitors(CLIENT, city)
    density = estimate_density(city)
    categories = categories_repartition(city)
    venues = DB.venue.find({'city': city, 'closed': {'$ne': True},
                            'cat': {'$ne': None}}, {'cat': 1})
    chosen = [v['_id'] for v in venues
              if len(visits.get(v['_id'], [])) > 4 and
              len(np.unique(visitors.get(v['_id'], []))) > 1 and
              not is_event(v['cat'])]
    print("Chosen {} venues in {}.".format(len(chosen), city))
    info, _ = venues_info(chosen, visits, visitors, density, depth=1)
    numeric = np.zeros((len(info), 24), dtype=np.float32)
    numeric[:, :5] = np.array([info['likes'], np.log(info['users']),
                               np.log(info['checkins']), info['H'],
                               info['Den']]).T
    numeric[:, 5] = [1e5 * CATS.index(c) for c in info['cat']]
    numeric[:, :3] = stats.zscore(numeric[:, :3], ddof=1)
    print("Got basic fact about it.")
    progress = AnimatedProgressBar(end=len(info), width=120)
    for idx, vid in enumerate(info.index):
        print(vid, info.irow(idx)['name'])
        cat, focus, ratio = full_surrounding(vid, lvenues, lphotos, lcheckins)
        numeric[idx, 6:15] = cat
        numeric[idx, 15] = focus
        numeric[idx, 16] = ratio
        own_visits = visits[vid]
        numeric[idx, 17] = is_week_end_place(own_visits)
        daily_visits = xp.aggregate_visits(own_visits, 1, 4)[0]
        numeric[idx, 18:] = xp.to_frequency(daily_visits)
        progress + 1
        # progress.show_progress()
    numeric[:, 6:15] = stats.zscore(numeric[:, 6:15], ddof=1)
    sio.savemat(city+'_fv', {'v': numeric, 'c': categories,
                'i': np.array(chosen)}, do_compression=True)
    return numeric, categories


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

    msg = 'Asked for {} but get only {}'.format(len(vids), len(venues))
    assert len(vids) == len(venues), msg
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
    surround = xp.build_surrounding(DB.venue, city, likes=-1, checkins=1)
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


def smoothed_location(loc, center, radius, city, pmapping=None):
    """Return a list of weight (obtained by a 2D Gaussian with `radius`)
    corresponding to the relative distance of points in `loc` with
    `center`. If `pmapping` is not None, it's a dictionnary {id: 2dpos} and
    center must be an id."""
    if len(loc) == 0:
        return []
    if len(loc) == 1:
        return [1.0]
    if pmapping:
        assert len(center) == 2
        origins = np.tile(center, (len(loc), 1))
        ploc = np.vstack([pmapping[_] for _ in loc]) - origins
    else:
        project = cm.cities.GEO_TO_2D[city]
        if 'coordinates' in center:
            center = center['coordinates']
        center = project(reversed(center))[:2]
    smooth = multivariate_normal([0, 0], (radius/2.5)*np.eye(2))
    return smooth.pdf(ploc/20)/smooth.pdf([0, 0])


def full_surrounding(vid, vmapping, pmapping, cmapping, city=None, radius=350):
    """Return a list of photos, checkins and venues categories in a `radius`
    around `vid`, within `city`. The mappings are dict({id: 2dpos})"""
    self = DB.venue.find_one({'_id': vid}, {'loc': 1, 'city': 1})
    city, position = city or self['city'], self['loc']
    ball = {'$geometry': position, '$maxDistance': radius}
    cat_distrib = categories_repartition(city, vid, ball, vmapping)
    photos = CLIENT.world.photos.find({'hint': city, 'loc': {'$near': ball}},
                                      {'venue': 1, 'taken': 1})
    pids, pvenue, ptime = u.xzip(photos, ['_id', 'venue', 'taken'])
    checkins = DB.checkin.find({'city': city, 'loc': {'$near': ball}},
                               {'time': 1, 'loc': 1})
    cids, ctime = u.xzip(checkins, ['_id', 'time'])
    center = vmapping[vid]
    focus = photo_focus(vid, center, pids, pvenue, radius, pmapping)
    photogeny = photo_ratio(center, pids, cids, radius, pmapping, cmapping)
    return cat_distrib, focus, photogeny


def photo_focus(vid, center, pids, pvenue, radius, mapping):
    """Return the ratio of photos with venue id around `vid` that are indeed
    about it."""
    this_venue = 0
    all_venues = 0
    smoothed = smoothed_location(pids, center, radius, None, mapping)
    for pid, weight in zip(pvenue, smoothed):
        if pid:
            if pid == vid:
                this_venue += weight
            else:
                all_venues += weight
    return 0 if all_venues < 1e-4 else this_venue / all_venues


def photo_ratio(center, pids, cids, radius, pmapping, cmapping):
    """Return log(nb_photos/nb_checkins) around `vid`, weighted by
    Gaussian."""
    p_smoothed = smoothed_location(pids, center, radius, None, pmapping)
    c_smoothed = smoothed_location(cids, center, radius, None, cmapping)
    # sum of c_smoothed because for the venue to exist, there must be some
    # checkins around.
    return np.log(np.sum(p_smoothed)/np.sum(c_smoothed))


def is_week_end_place(place_visits):
    """Tell if a place is more visited during the weekend."""
    is_we_visit = lambda h, d: d == 5 or (d == 4 and h >= 20) or \
        (d == 6 and h <= 20)
    we_visits = [1 for v in place_visits if is_we_visit(v.hour, v.weekday())]
    return int(len(we_visits) > 0.5*len(place_visits))


def categories_repartition(city, vid=None, ball=None, vmapping=None):
    """Return the distribution of top level Foursquare categories in
    `ball` (ie around `vid`) (or the whole `city` without weighting if
    None)."""
    if ball:
        radius = ball['$maxDistance']
        geo = {'$near': ball}
    else:
        geo = {'$ne': None}
    neighbors = DB.venue.find({'city': city, 'loc': geo},
                              {'cat': 1, 'cats': 1, 'loc': 1})
    vids, vcats = zip(*[(v['_id'], [v['cat']] + v['cats'])
                        for v in neighbors])
    smoothed_loc = itertools.cycle([1.0])
    if ball:
        smoothed_loc = smoothed_location(vids, vmapping[vid], radius, city,
                                         vmapping)
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
    raise ValueError('`kind` argument is not valid')


def draw_classes(centroid, offset, chunk=3):
    """Plot each time patterns in `centroid`."""
    size = centroid.shape[0]
    for i, marker in zip(range(size), LEGEND[:size]):
        ppl.plot(centroid[i, :], marker+'-', ms=9, c=ppl.colors.set1[i])
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
    import doctest
    doctest.testmod()
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
    describe_city(city)
