#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Interactive exploration of data file."""
import codecs
from collections import OrderedDict, defaultdict
from math import log
import scipy.io as sio
import numpy as np
import persistent
from more_query import get_top_tags
import CommonMongo as cm
import FSCategories as fsc


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
            return fsc.ID_TO_NAMES[path[depth]]
        return fsc.ID_TO_NAMES[path[-1]]

    summary = defaultdict(lambda: (0, 0))
    for venue in res:
        if venue['_id'] is not None:
            cat = parenting_cat(venue, depth)
            count, like = venue['count'], venue['like']
            summary[cat] = (summary[cat][0] + count, summary[cat][1] + like)

    return OrderedDict(sorted(summary.items(), key=lambda u: u[1][0]))


if __name__ == '__main__':
    #pylint: disable=C0103
    db, client = cm.connect_to_db('foursquare')
    checkins = db['checkin']
    # hourly, weekly, monthly = venues_activity(checkins, 'newyork', 15)
    ny_venue = describe_venue(db['venue'], 'newyork')
    print(ny_venue.items())
