#! /usr/bin/python2
# vim: set fileencoding=utf-8
import more_query as mq
import logging
import datetime
from multiprocessing import Pool, cpu_count
logging.basicConfig(filename='cleaning.log', level=logging.INFO,
                    format=u'%(asctime)s [%(levelname)s]: %(message)s')
T = 120
regions, to_region, _ = mq.k_split_bbox(mq.SF_BBOX, 200)
sot = datetime.datetime(2008, 1, 1)
now = datetime.datetime.utcnow()
time_step = 14*24*3600

def to_timecell(date):
    return int(mq.total_seconds(date - sot)/time_step) + 1


def clean_tags():
    users = mq.get_user_status(True).items()
    active_users = [u for u, c in users if c[0] >= T]
    logging.info('users considered: {}'.format(len(active_users)))
    pool = Pool(5)
    return sum(map(remove_tags, active_users))


def remove_tags(u):
    tags_used = DB.photos.aggregate([
        {'$match': {'hint': 'sf', 'uid': u}},
        {'$unwind': '$tags'},
        {'$group': {'_id': '$tags', 'count': {'$sum': 1}}}
    ])['result']
    active_tags = [t['_id'] for t in tags_used if t['count'] >= T]
    logging.info(u'{} has {} active tags'.format(u, len(active_tags)))
    removed = 0
    for t in active_tags:
        photos = mq.tag_location(DB.photos, t, None,
                                 sot, now, ['_id', 'taken'], user=u)
        count = [[] for dummy in range(len(regions)*to_timecell(now))]
        for p in photos:
            count[to_timecell(p[3])*(to_region(p[:2]) + 1)].append(p[2])
        to_pull = [p for ids in count if len(ids) >= T for p in ids]
        if len(to_pull) == 0:
            continue
        DB.photos.update({'_id': {'$in': to_pull}},
                         {'$pull': {'ntags': t}}, multi=True)
        removed += len(to_pull)
        logging.info(u'from {}, remove {} in {} photos.'.format(u, t,
                                                                len(to_pull)))

    return removed


if __name__ == '__main__':
    import pymongo
    client = pymongo.MongoClient('localhost', 27017)
    DB = client['flickr']
    mq.DB = DB
    tt = mq.clock()
    logging.info('removed tags in {} photos ({:.2f}).'.format(clean_tags(), mq.clock() - tt))
