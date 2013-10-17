#! /usr/bin/python2
# vim: set fileencoding=utf-8
import pymongo
from bson.son import SON
from bson.code import Code
from timeit import default_timer as clock


def last_query_time(db):
    return db.system.profile.find().limit(1).sort('ts', 1)[0]['millis']


if __name__ == '__main__':
    client = pymongo.MongoClient('localhost', 27017)
    db = client['flickr']
    photos = db['photos']
    start = clock()
    tags = photos.aggregate([
        {"$unwind": "$tags"},
        {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
        {"$sort": SON([("count", -1), ("_id", -1)])}
    ])
    t = 1000*(clock() - start)
    print('aggregate in {:.3f}ms ({})'.format(t, tags['result'][0]))
    mapper = Code("""
                  function () {
                      this.tags.forEach(function(z) {
                          emit(z, 1);
                     });
                 }""")
    reducer = Code("""
                   function (key, values) {
                       var total = 0;
                       for (var i = 0; i < values.length; i++) {
                           total += values[i];
                       }
                       return total;
                   }
                   """)
    res = photos.map_reduce(mapper, reducer, "myresults", full_response=True)
    t = res['timeMillis']
    tag = db.myresults.find().sort('value', -1)[0]
    print('mapRed in {}ms ({})'.format(t, tag))
