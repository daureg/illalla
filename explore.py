#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Interactive exploration of data file."""
import codecs
from collections import OrderedDict, defaultdict, namedtuple
from math import log
import scipy.io as sio
import scipy.spatial as spatial
import numpy as np
import persistent
from more_query import get_top_tags
import CommonMongo as cm
import FSCategories as fsc
Surrounding = namedtuple('Surrounding', ['tree', 'venues', 'id_to_index'])


def increase_coverage(upto=5000):
    """Save `upto` unprocessed San Francisco tags"""
    sup = persistent.load_var('supported')
    more = get_top_tags(upto, 'nsf_tag.dat')
    already = [v[0] for v in sup]
    addition = set(more).difference(set(already))
    persistent.save_var('addition', addition)


def read_entropies(grid=200, div=False):
    """Return a sorted dict of (tag, entropy or KL divergence)"""
    filename = 'n{}entropies_{}.dat'.format('K' if div else '', grid)
    with codecs.open(filename, 'r', 'utf8') as entropy:
        lines = [i.strip().split() for i in entropy.readlines()[1:]]
    entropies = sorted([(tag, float(val)) for val, tag in lines
                        if tag != '_background' and float(val) > 1e-5],
                       key=lambda x: x[1])
    return OrderedDict(entropies)


def spits_latex_table(N=10):
    N += 1
    e = []
    k = []
    for grid in [200, 80, 20]:
        tmp = read_entropies(grid)
        print(grid, max(tmp.values()), 2*log(grid))
        e.append(tmp.items()[:N])
        e.append(tmp.items()[-N:])
        tmp = read_entropies(grid, True)
        k.append(tmp.items()[:N])
        k.append(tmp.items()[-N:])
    line = u'{} & {:.3f} & {} & {:.3f} & {} & {:.3f} \\\\'
    # for i in range(N):
    #     print(line.format(e[0][i][0], e[0][i][1]/(2*log(200)),
    #                       e[2][i][0], e[2][i][1]/(2*log(80)),
    #                       e[4][i][0], e[4][i][1]/(2*log(20))))
    # for i in range(N):
    #     print(line.format(e[1][i][0], e[1][i][1]/(2*log(200)),
    #                       e[3][i][0], e[3][i][1]/(2*log(80)),
    #                       e[5][i][0], e[5][i][1]/(2*log(20))))
    for i in range(N-1, -1, -1):
        print(line.format(k[1][i][0], k[1][i][1]/get_max_KL(200),
                          k[3][i][0], k[3][i][1]/get_max_KL(80),
                          k[5][i][0], k[5][i][1]/get_max_KL(20)))
    for i in range(N-1, -1, -1):
        print(line.format(k[0][i][0], k[0][i][1]/get_max_KL(200),
                          k[2][i][0], k[2][i][1]/get_max_KL(80),
                          k[4][i][0], k[4][i][1]/get_max_KL(20)))


def get_max_KL(grid=200):
    """Return maximum KL divergence with size `grid`."""
    filename = 'freq_{}__background.mat'.format(grid)
    count = sio.loadmat(filename).values()[0]
    return -log(np.min(count[count > 0])/float(np.sum(count)))


def disc_latex(N=11):
    line = u'{} & {:.3f} & {} & {:.3f} & {} & {:.3f} \\\\'
    import persistent
    from rank_disc import top_discrepancy
    t = [persistent.load_var('disc/all'),
         persistent.load_var('disc/all_80'),
         persistent.load_var('disc/all_20')]
    supported = [v[0] for v in persistent.load_var('supported')]
    d = zip(*[top_discrepancy(l, supported) for l in t])
    display = lambda v: line.format(v[0][2], v[0][0], v[1][2], v[1][0],
                                    v[2][2], v[2][0])
    for v in d[:N]:
        print(display(v))
    for v in d[-N:]:
        print(display(v))


def venues_activity(checkins, city, limit=None):
    """Return time pattern of all the venues in 'city', or only the 'limit'
    most visited."""
    query = cm.build_query(city, True, ['lid', 'time'], limit)
    group = {'_id': '$lid', 'count': {'$sum': 1}, 'visits': {'$push': '$time'}}
    query.insert(2, {'$group': group})
    if isinstance(limit, int) and limit > 0:
        query.insert(-1, {'$sort': {'count': -1}})
    res = checkins.aggregate(query)['result']
    hourly = []
    weekly = []
    # monthly pattern may not be that relevant since the dataset does not cover
    # a whole year
    monthly = []
    for venue in res:
        timing = np.array([(t.hour, t.weekday(), t.month)
                           for t in venue['visits']])
        hourly.append(list(np.bincount(timing[:, 0], minlength=24)))
        weekly.append(list(np.bincount(timing[:, 1], minlength=7)))
        monthly.append(list(np.bincount(timing[:, 2], minlength=12)))
    return hourly, weekly, monthly


def describe_venue(venues, city, depth=2, limit=None):
    """Gather some statistics about venue, aggregating categories at `depth`
    level."""
    query = cm.build_query(city, False, ['cat', 'likes'], limit)
    group = {'_id': '$cat', 'count': {'$sum': 1}, 'like': {'$sum': '$likes'}}
    query.extend([{'$group': group}, {'$sort': {'count': -1}}])
    cats = fsc.get_categories()
    res = venues.aggregate(query)['result']

    def parenting_cat(place, depth):
        """Return the category of `place`, without going beyond `depth`"""
        _, path = fsc.search_categories(cats, place['_id'])
        if len(path) > depth:
            return fsc.CAT_TO_ID[:path[depth]]
        return fsc.CAT_TO_ID[:path[-1]]

    summary = defaultdict(lambda: (0, 0))
    nb_venues = 0
    for venue in res:
        if venue['_id'] is not None:
            cat = parenting_cat(venue, depth)
            count, like = venue['count'], venue['like']
            nb_venues += count
            summary[cat] = (summary[cat][0] + count, summary[cat][1] + like)

    for cat, stat in summary.iteritems():
        count, like = stat
        summary[cat] = (100.0*count/nb_venues, count, like)
    return OrderedDict(sorted(summary.items(), key=lambda u: u[1][0],
                              reverse=True))


def build_surrounding(venues, city):
    """Return a scipy backed 2-d tree of all venues in `city` with their
    categories."""
    assert city in cm.cities.SHORT_KEY, 'not a valid city'
    res = list(venues.find({'city': city, 'likes': {'$gt': 0},
                            'checkinsCount': {'$gte': 10}},
                           {'cat': 1, 'loc.coordinates': 1}))
    indexing = fsc.bidict.bidict()
    places = np.zeros((len(res), 3))  # pylint: disable=E1101
    for pos, venue in enumerate(res):
        numeric_category = fsc.ID_TO_INDEX[venue['cat']]
        lng, lat = venue['loc']['coordinates']
        local_coord = cm.cities.GEO_TO_2D[city]([lat, lng])
        places[pos, :] = (local_coord[0], local_coord[1], numeric_category)
        indexing[pos] = venue['_id']
    # pylint: disable=E1101
    return Surrounding(spatial.KDTree(places[:, :2]), places, indexing)


def query_surrounding(surrounding, venue_id, radius=150):
    """Return the venues in `surrounding` closer than `radius` from
    `venue_id`."""
    from_index = lambda idx: surrounding.id_to_index[idx]
    to_index = lambda vid: surrounding.id_to_index[:vid]
    queried_index = to_index(venue_id)
    full_venue = surrounding.venues[queried_index]
    position = full_venue[:2]
    neighbors = surrounding.tree.query_ball_point(position, radius)
    return [from_index(i) for i in neighbors if i is not queried_index]


def alt_surrounding(venues_db, venue_id, radius=150):
    position = venues_db.find_one({'_id': venue_id}, {'loc': 1})['loc']
    ball = {'$geometry': position, '$maxDistance': radius}
    neighbors = venues_db.find({'city': 'helsinki', 'loc': {'$near': ball},
                                'likes': {'$gt': 0},
                                'checkinsCount': {'$gte': 10}},
                               {'cat': 1, 'time': 1})
    return [v['_id'] for v in neighbors if v['_id'] != venue_id]

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    #pylint: disable=C0103
    db, client = cm.connect_to_db('foursquare')
    checkins = db['checkin']
    city = 'barcelona'
    # hourly, weekly, monthly = venues_activity(checkins, 'newyork', 15)
    stats = lambda s: '{:.2f}% of checkins ({}), {} likes'.format(*s)
    ny_venue = describe_venue(db['venue'], city, 1)
    with codecs.open(city + '_1_cat.dat', 'w', 'utf8') as report:
        report.write(u'\n'.join([u'{}: {}'.format(k, stats(v))
                                 for k, v in ny_venue.items()]))
    ny_venue = describe_venue(db['venue'], city, 2)
    with codecs.open(city + '_2_cat.dat', 'w', 'utf8') as report:
        report.write(u'\n'.join([u'{}: {}'.format(k, stats(v))
                                 for k, v in ny_venue.items()]))
    # surround = build_surrounding(db['venue'], 'helsinki')
    # a = set(query_surrounding(surround, '4c619433a6ce9c74ba5ef1d6', 70))
    # b = set(alt_surrounding(db['venue'], '4c619433a6ce9c74ba5ef1d6', 70))
    # print(b-a)
