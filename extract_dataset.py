#! /usr/bin/python2
# vim: set fileencoding=utf-8
import codecs
import CommonMongo as cm
from timeit import default_timer as clock
from bson.son import SON
from more_query import FIRST_TIME, LAST_TIME, compute_entropy
from collections import Counter
from persistent import save_var, load_var
import scipy.sparse
import numpy as np
import scipy.io as sio
import calendar
DAY_STEP = 4


def write_tagset(DB):
    cursor = DB.photos.find({}, {'_id': 0, 'tags': 1}).limit(50)
    tags = [u'\t'.join(p['tags']) for p in cursor]
    with codecs.open('dataset.txt', 'w', 'utf8') as f:
        f.write(u'\n'.join(tags))


def supported_tags(DB, city, photos_threshold=150, users_threshold=25, timespan=500):
    interval = timespan*24*3600
    return [(p['_id'], p['count'], len(p['users']), p['first'], p['last'])
            for p in DB.photos.aggregate([
                {"$match": {"hint": city}},
                {'$project': {'_id': 0, 'upload': 1,
                              'user': '$uid', 'tags': 1}},
                {"$unwind": "$tags"},
                {'$group': {'_id': '$tags',
                            'first': {'$min': '$upload'},
                            'last': {'$max': '$upload'},
                            'users': {'$addToSet': '$user'},
                            "count": {"$sum": 1}}},
                # http://stackoverflow.com/a/15224544 for the exist trick
                {"$match": {"count": {"$gte": photos_threshold},
                            "users."+str(users_threshold-1): {"$exists": 1}}},
                # {"$sort": SON([("count", -1)])}
            ])['result']
            # ]
            if (p['last'] - p['first']).total_seconds() > interval]


def tag_time(DB, tag, fm=None):
    query = {}
    field = {'taken': 1, '_id': 0, 'uid': 1, 'ntags': 1, 'loc': 1}
    query['hint'] = 'sf'
    if isinstance(tag, list):
        query['ntags'] = {'$in': tag}
    else:
        query['ntags'] = tag
    query['taken'] = {'$gte': FIRST_TIME, '$lte': LAST_TIME}
    cursor = DB.photos.find(query, field)

    def format_photo(p):
        if fm is None:
            return (p['taken'].weekday(), p['taken'].hour/DAY_STEP)
        return fm(p)

    return [format_photo(p) for p in cursor]


def period_entropy(DB, tag):
    times = [p[0] + DAY_STEP*p[1] for p in tag_time(DB, tag)]
    return float(compute_entropy(Counter(times)))


def get_data(DB):
    entropies = load_var('Hsupported')
    tags = sorted([k for k, v in entropies.items() if 2.5 <= v <= 3.01])
    save_var('mat_tag', tags)
    u = load_var('user_status')
    user_index = {k: i for i, k in enumerate(u)}

    def format_photo(p):
        user = user_index[p['uid']]
        loc = p['loc']['coordinates']
        taken = [p['taken'].weekday(), p['taken'].hour,
                 calendar.timegm(p['taken'].utctimetuple())]
        indicator = [int(t in p['ntags']) for t in tags]
        return [user] + loc + taken + indicator

    photos_feature = np.mat(tag_time(DB, tags, format_photo))
    sio.savemat('deep', {'A': scipy.sparse.csr_matrix(photos_feature)})

if __name__ == '__main__':
    import arguments
    args = arguments.city_parser().parse_args()
    city = args.city
    DB, client = cm.connect_to_db('world', args.host, args.port)
    s = clock()
    tags = supported_tags(DB, city, photos_threshold=30, users_threshold=5,
                          timespan=60)
    save_var(city+'_tag_support', tags)
    # entropies = {t[0]: period_entropy(DB, t[0]) for t in tags}
    # save_var('Hsupported', entropies)
    # get_data(DB)
    t = clock()
    print(t-s)
