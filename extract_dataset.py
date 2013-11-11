#! /usr/bin/python2
# vim: set fileencoding=utf-8
import codecs
import pymongo
from timeit import default_timer as clock
from bson.son import SON
from more_query import FIRST_TIME, LAST_TIME, compute_entropy
from collections import Counter
from persistent import save_var
DAY_STEP = 4


def write_tagset(DB):
    cursor = DB.photos.find({}, {'_id': 0, 'tags': 1}).limit(50)
    tags = [u'\t'.join(p['tags']) for p in cursor]
    with codecs.open('dataset.txt', 'w', 'utf8') as f:
        f.write(u'\n'.join(tags))


def supported_tags(DB, photos_threshold=150, users_threshold=25, timespan=500):
    interval = timespan*24*3600
    return [(p['_id'], p['count'], len(p['users']))
            for p in DB.photos.aggregate([
                {"$match": {"hint": "sf"}},
                {'$project': {'_id': 0, 'upload': 1,
                              'user': '$uid', 'ntags': 1}},
                {"$unwind": "$ntags"},
                {'$group': {'_id': '$ntags',
                            'first': {'$min': '$upload'},
                            'last': {'$max': '$upload'},
                            'users': {'$addToSet': '$user'},
                            "count": {"$sum": 1}}},
                # http://stackoverflow.com/a/15224544 for the exist trick
                {"$match": {"count": {"$gte": photos_threshold},
                            "users."+str(users_threshold-1): {"$exists": 1}}},
                {"$sort": SON([("count", -1)])}
            ])['result']
            if (p['last'] - p['first']).total_seconds() > interval]


def tag_time(DB, tag):
    query = {}
    field = {'taken': 1, '_id': 0}
    query['hint'] = 'sf'
    query['ntags'] = tag
    query['taken'] = {'$gte': FIRST_TIME, '$lte': LAST_TIME}
    cursor = DB.photos.find(query, field)
    return [(p['taken'].weekday(), p['taken'].hour/DAY_STEP) for p in cursor]


def period_entropy(DB, tag):
    times = [p[0] + DAY_STEP*p[1] for p in tag_time(DB, tag)]
    return float(compute_entropy(Counter(times)))


if __name__ == '__main__':
    client = pymongo.MongoClient('localhost', 27017)
    DB = client['flickr']
    s = clock()
    tags = supported_tags(DB)
    save_var('supported', tags)
    entropies = {t[0]: period_entropy(DB, t[0]) for t in tags}
    save_var('Hsupported', entropies)
    t = clock()
    print(t-s)
