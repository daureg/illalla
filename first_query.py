#! /usr/bin/python2
# vim: set fileencoding=utf-8
import pymongo
from bson.son import SON
from timeit import default_timer as clock
from operator import itemgetter
from outplot import outplot
from more_query import inside_bbox
try:
    from collections import OrderedDict
except ImportError:
    from OrderedDict import OrderedDict


def last_query_time(db):
    return db.system.profile.find().sort('ts', -1).limit(1)[0]['millis']


def tag_count(mongo, city, flickr=True, bbox=None, output='tags_count.dat'):
    """Return ordered {tags: count} dict from `flickr` photos (or Foursquare
    venues), in `city` and optionally within `bbox`."""
    start = clock()
    city_field = 'hint' if flickr else 'city'
    collec = mongo.world.photos if flickr else mongo.foursquare.venue
    query = [{"$match": {city_field: city}}, {"$unwind": "$tags"},
             {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
             {"$sort": SON([("count", -1), ("_id", -1)])}]
    if bbox:
        query[0]['$match'].update({"loc": inside_bbox(bbox)})
    tags = collec.aggregate(query)
    t = 1000*(clock() - start)
    print('aggregate in {:.3f}ms ({})'.format(t, tags['result'][0]))
    name = map(itemgetter('_id'), tags['result'])
    count = map(itemgetter('count'), tags['result'])

    outplot(output, ['tag', 'count'], name, count)
    return OrderedDict(zip(name, count))


def user_loc(db):
    start = clock()
    # users = photos.aggregate([
    #     {'$project': {'_id': 0, 'user': '$uid'}},
    #     {'$group': {'_id': '$user', 'photos': {'$sum': 1}}},
    #     {'$sort': SON([('photos', -1)])}
    # ])
    users = photos.aggregate([
        {'$project': {'_id': 0, 'loc.coordinates': 1, 'user': '$uid'}},
        {'$group': {'_id': '$user', 'pos': {'$push': '$loc.coordinates'}}},
        {'$limit': 5}
    ])
    t = 1000*(clock() - start)
    print('aggregate in {:.3f}ms ({})'.format(t, users['result'][0]))
    return users

if __name__ == '__main__':
    client = pymongo.MongoClient('localhost', 27017)
    db = client['flickr']
    photos = db['photos']
    # db.set_profiling_level(pymongo.ALL)
    # SF_BBOX = [37.7123, -122.531, 37.84, -122.35]
    tc = tag_count(client, 'paris', flickr=False, output='paris_fs_tag.dat')
    # u, c = user_count(photos)
    # u = user_loc(photos)
