#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Interactive exploration of data file."""
import codecs
from collections import OrderedDict, defaultdict, namedtuple
from math import log
import scipy.io as sio
import scipy.spatial as spatial
import scipy.cluster.vq as cluster
import numpy as np
import persistent
import CommonMongo as cm
import FSCategories as fsc
import AskFourquare as af
Surrounding = namedtuple('Surrounding', ['tree', 'venues', 'id_to_index'])
from utils import human_day, answer_to_dict, geodesic_distance
import itertools
import enum
Entity = enum.Enum('Entity', 'checkin venue user photo')
from random import sample


def increase_coverage(upto=5000):
    """Save `upto` unprocessed San Francisco tags"""
    from more_query import get_top_tags
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


def get_visitors(mongo, city=None, ball=None):
    """Return a sequence of [user] for each venue within `city` or `ball` =
    ((lng, lat), radius) by querying a `mongo` client."""
    operation, _ = choose_query_type(mongo, Entity.venue)
    location = get_spatial_query(Entity.venue, city, ball)
    return query_for_visits(operation, location, 'tuid', mongo, city)


def get_visits(mongo, entity, city=None, ball=None):
    """Return a sequence of [timestamp] for each `entity` (venue or photo)
    within `city` or `ball` = ((lng, lat), radius) by querying a `mongo`
    client."""
    operation, time = choose_query_type(mongo, entity)
    location = get_spatial_query(entity, city, ball)
    return query_for_visits(operation, location, time, mongo, city)


def choose_query_type(mongo, entity):
    """Return appropriate db operation and time field name."""
    if entity == Entity.venue:
        return mongo.foursquare.checkin.aggregate, 'time'
    elif entity == Entity.photo:
        return mongo.world.photos.find, 'taken'
    else:
        raise ValueError('choose venue or photo')


def get_spatial_query(kind, city, ball):
    """Return appropriate mongo geographical query operator."""
    if city and city in cm.cities.SHORT_KEY:
        return {('city' if kind == Entity.venue else 'hint'): city,
                'lid': {'$ne': None}}
    elif ball and len(ball) == 2:
        center, radius = ball
        center = {'type': 'Point', 'coordinates': list(center)}
        ball = {'$geometry': center, '$maxDistance': radius}
        return {'loc': {'$near': ball}}
    else:
        raise ValueError('choose city or ball')


def query_for_visits(operation, location, time, mongo, city):
    """Return a dict resulting of the call of `operation` with `location` and
    `time` arguments."""
    if 'find' in str(operation.im_func):
        return answer_to_dict(operation(location, {time: 1}))
    # $near is not supported in aggregate $match
    if 'loc' in location:
        ids = mongo.foursquare.venue.find({'city': city,
                                           'loc': location['loc']}, {'_id': 1})
        ids = [v['_id'] for v in ids]
        location['lid'] = {'$in': ids}
        del location['loc']
    match = {'$match': location}
    project = {'$project': {'time': '$'+time, 'lid': 1, '_id': 0}}
    group = {'$group': {'_id': '$lid', 'visits': {'$push': '$time'}}}
    query = [match, project, group]
    convert = (lambda x: map(int, x)) if time == 'tuid' else None
    return answer_to_dict(itertools.chain(operation(query)['result']), convert)


import Chunker
def collapse(values, chunk_size, offset=0):
    """Return sum of `values` by piece of `chunk_size` (starting from `offset`
    and then cycling).
    >>> collapse(range(6), 3)
    array([ 3, 12])
    >>> collapse(range(8), 2, 2)
    array([ 5,  9, 13,  1])
    >>> collapse([4, 0, 0, 0, 0, 2], 3, 1)
    array([0, 6])
    >>> collapse(range(6), 2, 1)
    array([3, 7, 5])"""
    values = list(values.ravel())
    length = len(values)
    assert length % chunk_size == 0, 'there will be leftovers'
    # pylint: disable=E1101
    chunker = Chunker.Chunker(chunk_size)
    return np.array([sum(chunk)
                     for chunk in chunker(values[offset:]+values[:offset])])


def aggregate_visits(visits, offset=0, chunk=3, weights=None):
    """Transform a list of visits into hourly and daily pattern (grouping
    hours by chunk of size `chunk`, starting from `offset`)."""
    # pylint: disable=E1101
    histo = lambda dim, size: np.bincount(timing[:, dim], weights,
                                          minlength=size)
    timing = np.array([(v.hour, human_day(v)) for v in visits])
    return collapse(histo(0, 24), chunk, offset), histo(1, 7*3)


def to_frequency(data):
    """Take a list of lists and return the corresponding frequency matrix."""
    #TODO handle division by 0
    # pylint: disable=E1101
    if hasattr(data, 'shape') and len(data.shape) == 1:
        return data / np.sum(data, dtype=np.float)
    totals = np.sum(data, 1)
    nb_lines = len(data[0])
    return data/np.tile(np.array([totals], dtype=np.float).T, (1, nb_lines))


def clusterize(patterns):
    """try to find the best k by running k means on pattern."""
    whitened = cluster.whiten(patterns)
    distorsion = [cluster.kmeans(whitened, i) for i in range(2, 24)]
    return distorsion


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
    for venue in res:
        hour, day = aggregate_visits(venue['visits'])
        hourly.append(hour)
        weekly.append(day)
    return hourly, weekly


def describe_venue(venues, city, depth=2, limit=None):
    """Gather some statistics about venue, aggregating categories at `depth`
    level."""
    query = cm.build_query(city, False, ['cat', 'likes'], limit)
    group = {'_id': '$cat', 'count': {'$sum': 1}, 'like': {'$sum': '$likes'}}
    query.extend([{'$group': group}, {'$sort': {'count': -1}}])
    res = venues.aggregate(query)['result']

    def parenting_cat(place, depth):
        """Return the category of `place`, without going beyond `depth`"""
        _, path = fsc.search_categories(place['_id'])
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


def build_surrounding(venues, city, likes=5, checkins=30):
    """Return a scipy backed 2-d tree of venues in `city` with their
    categories, provided that they have enough `likes` and `checkins`."""
    assert city in cm.cities.SHORT_KEY, 'not a valid city'
    res = list(venues.find({'city': city, 'likes': {'$gt': likes},
                            'checkinsCount': {'$gte': checkins}},
                           {'cat': 1, 'loc.coordinates': 1}))
    indexing = fsc.bidict.bidict()
    places = np.zeros((len(res), 3))  # pylint: disable=E1101
    for pos, venue in enumerate(res):
        numeric_category = 0 if not venue['cat'] else fsc.ID_TO_INDEX[venue['cat']]
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


def collect_similars(venues_db, client, city):
    """Find similars venues for 100 location in city, save the result in DB and
    return matching venues that were already in DB."""
    venues = answer_to_dict(venues_db.find({'city': city}, {'loc': 1}))
    chosen = sample(venues.items(), 500)
    distances = []
    all_match = []
    for vid, loc in chosen:
        similars = af.similar_venues(vid, client=client)
        if similars is None:
            continue
        else:
            print(vid, similars)
        venues_db.update({'_id': vid}, {'$set': {'similars': similars}})
        matching = answer_to_dict(venues_db.find({'_id': {'$in': similars}},
                                                 {'loc': 1}))
        all_match.append(matching)
        distances.append([geodesic_distance(loc, sloc)
                          for sloc in matching.itervalues()])
    return chosen, distances, all_match


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    # pylint: disable=C0103
    import arguments
    args = arguments.city_parser().parse_args()
    city = args.city
    db, client = cm.connect_to_db('foursquare', args.host, args.port)
    checkins = db['checkin']
