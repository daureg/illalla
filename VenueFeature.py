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
import scipy.io as sio
import scipy.cluster.vq as cluster
try:
    from scipy.stats import multivariate_normal
except ImportError:
    from _multivariate import multivariate_normal
import re
import string
import persistent as p
import Surrounding as s
NOISE = re.compile(r'[\s'+string.punctuation+r']')
DB = None
CLIENT = None
LEGEND = 'v^<>s*xo|8d+'
CATS = ['Arts & Entertainment', 'College & University', 'Food',
        'Nightlife Spot', 'Outdoors & Recreation', 'Shop & Service',
        'Professional & Other Places', 'Residence', 'Travel & Transport']
# top_cats = [top_cat.name for top_cat in CATS.sub if top_cat.name != 'Event']
# cats2 = {sub_cat.name: int(1e5)*top_cats.index(top_cat.name)+idx+1
#          for top_cat in CATS.sub if top_cat.name != 'Event'
#          for idx, sub_cat in enumerate(top_cat.sub)}
# p.save_var('cat_depth_2.my', cats2)
TOP_CATS = {None: None}
# TOP_CATS.update({_: parenting_cat(_)
#                  for _ in fsc.get_subcategories('1')[1:]})
RADIUS = 350
SMOOTH = multivariate_normal([0, 0], (RADIUS/2.5)*np.eye(2))
SMOOTH_MAX = SMOOTH.pdf([0, 0])


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


def global_info(city, standalone=False):
    """Gather global statistics about `city`."""
    lvenues = geo_project(city, DB.venue.find({'city': city}, {'loc': 1}))
    lcheckins = geo_project(city, DB.checkin.find({'city': city}, {'loc': 1}))
    lphotos = geo_project(city, CLIENT.world.photos.find({'hint': city},
                                                         {'loc': 1}))
    local_projection = [lvenues, lcheckins, lphotos]
    visits = xp.get_visits(CLIENT, xp.Entity.venue, city)
    visitors = xp.get_visitors(CLIENT, city)
    density = estimate_density(city)
    activity = [visits, visitors, density]
    global TOP_CATS
    TOP_CATS = p.load_var('top_cats.my')
    infos = {'venue': [] if standalone else ['cat', 'cats'],
             'photo': ['taken'] if standalone else ['venue']}
    svenues = s.Surrounding(DB.venue, {'city': city}, infos['venue'], lvenues)
    scheckins = s.Surrounding(DB.checkin, {'city': city}, ['time'], lcheckins)
    sphotos = s.Surrounding(CLIENT.world.photos, {'hint': city},
                            infos['photo'], lphotos)
    surroundings = [svenues, scheckins, sphotos]
    p.save_var('{}_s{}s.my'.format(city, 'venue'), svenues)
    if standalone:
        for name, var in zip(['venue', 'checkin', 'photo'], surroundings):
            p.save_var('{}_s{}s.my'.format(city, name), var)
    return local_projection + activity + surroundings


def describe_city(city):
    """Compute feature vector for selected venue in `city`."""
    CATS2 = p.load_var('cat_depth_2.my')
    # a few venues don't have level 2 categories (TODO add it manually?)
    CATS2.update({cat: int(idx*1e5) for idx, cat in enumerate(CATS)})
    info = global_info(city)
    lvenues, lcheckins, lphotos = info[:3]
    visits, visitors, density = info[3:6]
    nb_visitors = np.unique(np.array([v for place in visitors.itervalues()
                                      for v in place])).size
    svenues, scheckins, sphotos = info[6:]
    categories = categories_repartition(city, svenues, lvenues, RADIUS)
    venues = DB.venue.find({'city': city, 'closed': {'$ne': True},
                            'cat': {'$ne': None}, 'usersCount': {'$gt': 1}},
                           {'cat': 1})
    chosen = [v['_id'] for v in venues
              if len(visits.get(v['_id'], [])) > 4 and
              len(np.unique(visitors.get(v['_id'], []))) > 1 and
              not is_event(v['cat'])]
    print("Chosen {} venues in {}.".format(len(chosen), city))
    info, _ = venues_info(chosen, visits, visitors, density, depth=2,
                          tags_freq=False)
    print("{} of them will be in the matrix.".format(len(info)))
    numeric = np.zeros((len(info), 31), dtype=np.float32)
    numeric[:, :5] = np.array([info['likes'], info['users'], info['checkins'],
                               info['H'], info['Den']]).T
    print('venues with no level 2 category:')
    print([info.index[i] for i, c in enumerate(info['cat'])
           if CATS2[c] % int(1e5) == 0])
    numeric[:, 5] = [CATS2[c] for c in info['cat']]
    numeric[:, 24] = np.array(info['Ht'])
    for idx, vid in enumerate(info.index):
        surrounding = full_surrounding(vid, lvenues, lphotos, lcheckins,
                                       svenues, scheckins, sphotos, city)
        cat, focus, ratio, around_visits = surrounding
        numeric[idx, 6:15] = cat
        numeric[idx, 15] = focus
        numeric[idx, 16] = ratio
        own_visits = visits[vid]
        numeric[idx, 17] = is_week_end_place(own_visits)
        daily_visits = xp.aggregate_visits(own_visits, 1, 4)[0]
        numeric[idx, 18:24] = xp.to_frequency(daily_visits)
        numeric[idx, 25:31] = xp.to_frequency(around_visits)
    weird = np.argwhere(np.logical_or(np.isnan(numeric), np.isinf(numeric)))
    numeric[weird] = 0.0
    sio.savemat(city+'_fv', {'v': numeric, 'c': categories,
                             'i': np.array(list(info.index)),
                             'stat': [nb_visitors]}, do_compression=True)


def venues_info(vids, visits=None, visitors=None, density=None, depth=10,
                tags_freq=True):
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
    if tags_freq:
        res['tags'] = [[normalized_tag(t) for t in _['tags']] for _ in venues]
    loc = [_['loc']['coordinates'] for _ in venues]
    get_cat = lambda c, d: top_category(c) if d == 1 else parenting_cat(c, d)
    res['cat'] = [get_cat(_['cat'], depth) for _ in venues]
    res['vis'] = [len(visits[id_]) for id_ in res.index]
    res['H'] = [venue_entropy(visitors[id_]) for id_ in res.index]
    res['Ht'] = [time_entropy(visits[id_]) for id_ in res.index]
    coords = np.fliplr(np.array(loc))
    points = cm.cities.GEO_TO_2D[city](coords)
    res['Den'] = density(points)
    if tags_freq:
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


def smoothed_location(loc, center, radius, city, pmapping):
    """Return a list of weight (obtained by a 2D Gaussian with `radius`)
    corresponding to the relative distance of points in `loc` with
    `center`. `pmapping` is a dictionnary {id: 2dpos} and `center` a 2D
    point."""
    if len(loc) == 0:
        return []
    if len(loc) == 1:
        return [1.0]
    assert len(center) == 2
    # TODO: loc could directly be the subset
    ploc = np.array([pmapping[_] for _ in loc]) - center
    return SMOOTH.pdf(ploc/20)/SMOOTH_MAX


def full_surrounding(vid, vmapping, pmapping, cmapping, svenues, scheckins,
                     sphotos, city, radius=350):
    """Return a list of photos, checkins and venues categories in a `radius`
    around `vid`, within `city`. The mappings are dict({id: 2dpos})"""
    cat_distrib = categories_repartition(city, svenues, vmapping, radius, vid)
    center = vmapping[vid]
    pids, infos, _ = sphotos.around(center, radius)
    pvenue = infos[0]
    cids, infos, _ = scheckins.around(center, radius)
    ctime = infos[0]
    focus = photo_focus(vid, center, pids, pvenue, radius, pmapping)
    photogeny, c_smoothed = photo_ratio(center, pids, cids, radius, pmapping,
                                        cmapping)
    if len(ctime) < 5:
        print(vid + ' is anomalous because there is less than 5 check-in in a 350m radius')
    if len(ctime) == 0:
        surround_visits = np.ones(6)
    else:
        surround_visits = xp.aggregate_visits(ctime, 1, 4, c_smoothed)[0]
    return cat_distrib, focus, photogeny, surround_visits


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
    """Return nb_photos/nb_checkins around `vid`, weighted by Gaussian."""
    p_smoothed = smoothed_location(pids, center, radius, None, pmapping)
    c_smoothed = smoothed_location(cids, center, radius, None, cmapping)
    # sum of c_smoothed ≠ 0 because for the venue to exist, there must be some
    # checkins around. NOTE: actually, there are anomalous venues for which it
    # is not the case
    return np.sum(p_smoothed)/np.sum(c_smoothed), c_smoothed


def is_week_end_place(place_visits):
    """Tell if a place is more visited during the weekend."""
    is_we_visit = lambda h, d: d == 5 or (d == 4 and h >= 20) or \
        (d == 6 and h <= 20)
    we_visits = [1 for v in place_visits if is_we_visit(v.hour, v.weekday())]
    return int(len(we_visits) > 0.5*len(place_visits))


def categories_repartition(city, svenues, vmapping, radius, vid=None):
    """Return the distribution of top level Foursquare categories in
    `ball` (ie around `vid`) (or the whole `city` without weighting if
    None)."""
    smoothed_loc = itertools.cycle([1.0])
    if vid:
        vids, vcats, _ = svenues.around(vmapping[vid], radius)
        smoothed_loc = smoothed_location(vids, vmapping[vid], radius, city,
                                         vmapping)
    else:
        vids, vcats, _ = svenues.all()
    vcats = vcats[0]
    distrib = defaultdict(int)
    for own_cat, weight in zip(vcats, smoothed_loc):
        for cat in own_cat:
            distrib[TOP_CATS[cat]] += weight
    distrib = np.array([distrib[c] for c in CATS])
    # Can't be zero because there is always at least the venue itself in
    # surrounding.
    # TODO: maybe it would be more informative to return how it deviate from
    # the global distribution.
    return distrib / np.sum(distrib)


def venue_entropy(visitors):
    """Compute the entropy of venue given the list of its `visitors`."""
    # pylint: disable=E1101
    return u.compute_entropy(np.array(Counter(visitors).values(), dtype=float))


def time_entropy(visits):
    """Compute entropy of venue with respect to time of the day of its
    checkins."""
    hours = np.bincount([t.hour for t in visits], minlength=24)
    return u.compute_entropy(hours.astype(float))/np.log(24.0)


def normalized_tag(tag):
    """normalize `tag` by removing punctuation and space character."""
    return NOISE.sub('', tag).lower()


def count_tags(tags):
    """Count occurence of a list of list of tags."""
    return Counter([normalized_tag(t) for oneset in tags for t in oneset])


@u.memodict
def top_category(cat):
    return parenting_cat(cat, 1)


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
    import arguments
    args = arguments.city_parser().parse_args()
    city = args.city
    DB, CLIENT = cm.connect_to_db('foursquare', args.host, args.port)

    # pylint: disable=E1101
    do_cluster = lambda val, k: cluster.kmeans2(val, k, 20, minit='points')

    def getclass(c, kl, visits):
        """Return {id: time pattern} of the venues in class `c` of
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
    for c in cm.cities.SHORT_KEY:
        if c == 'newyork':
            continue
        describe_city(c)
    # describe_city(city)
    # for c in ['amsterdam', 'london', 'moscow', 'prague', 'stockholm']:
    #     global_info(c, standalone=False)
    # global_info(city, standalone=True)
    # lvenues = geo_project(city, DB.venue.find({'city': city}, {'loc': 1}))
    # svenues = s.Surrounding(DB.venue, {'city': city}, [], lvenues)
    # p.save_var('{}_s{}s.my'.format(city, 'venue'), svenues)
    # p.save_var('{}_l{}s.my'.format(city, 'venue'), lvenues)
