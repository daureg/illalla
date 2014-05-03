#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Read ICWSM 11 Foursquare dataset and keep only check in occuring in
specified cities.
In the following, x refers to the latitude and y to the longitude"""
import cities
import urlparse
import pymongo
from VenueIdCrawler import VenueIdCrawler
from collections import namedtuple, defaultdict
from datetime import datetime
import twitter_helper as th
from bisect import bisect_left
# from calendar import timegm

Point = th.Point
Location = namedtuple('Location', ['type', 'coordinates'])
CheckIn = namedtuple('CheckIn',
                     ['tid', 'lid', 'uid', 'city', 'loc', 'time', 'place'])
Node = th.Node
BLACKLIST = ['fst.je', 'gowal.la', 'picplz.com', 'wal.la', 'flic.kr',
             'myloc.me', 'wp.me', 'yfrog.com', 'j.mp', 'bkite.com', 'untpd.it']
MISSING_ID = []


def convert_checkin_for_mongo(checkin):
    suitable = checkin._asdict()
    # because namedtuple cannot have field starting with underscore.
    # TODO: Should I use namedtuple in the first place?
    suitable['_id'] = suitable['tid']
    del suitable['tid']
    return suitable


def save_to_mongo(documents, destination, venues_getter):
    urls = [c.lid for c in documents]
    ids = venues_getter.venue_id_from_urls(urls)
    converted = []
    for i, c in enumerate(documents):
        converted.append(convert_checkin_for_mongo(c))
        converted[-1]['lid'] = ids[i]
    try:
        destination.insert(converted, continue_on_error=True)
    except pymongo.errors.DuplicateKeyError:
        # it's not my dataset so there is no much I can do
        pass


def update_time_and_place(documents, destination):
    for checkin in documents:
        _id, time, place = checkin
        try:
            destination.update({'_id': _id},
                               {'$set': {'time': time, 'place': place}})
        except pymongo.errors.OperationFailure as err:
            print(err, err.coderr)


def id_must_be_process(_id):
    i = bisect_left(MISSING_ID, _id)
    return i != len(MISSING_ID) and MISSING_ID[i] == _id


def extract_url_from_msg(msg):
    last_word = msg.split()[-1]
    url = None
    if last_word.startswith('htt') and len(last_word) < 24:
        host = urlparse.urlparse(last_word).netloc
        if host in BLACKLIST:
            last_word = None
        url = last_word
    return url

if __name__ == '__main__':
    from persistent import save_var, load_var

    try:
        previously = load_var('avenues_id_new_kosh')
    except IOError:
        previously = None
    venues_getter = VenueIdCrawler(previously, use_network=False)

    checkins = None
    client = pymongo.MongoClient('localhost', 27017)
    db = client['foursquare']
    checkins = db['checkin']
    checkins.ensure_index([('loc', pymongo.GEOSPHERE),
                           ('lid', pymongo.ASCENDING),
                           ('city', pymongo.ASCENDING),
                           ('time', pymongo.ASCENDING)])
    import sys
    infile = 'verysmall' if len(sys.argv) < 2 else sys.argv[1]
    tree = th.obtain_tree()
    stats = defaultdict(lambda: 0)

    # def find_city(x, y):
    #     for city in bboxes:
    #         if city.contains(x, y):
    #             return city.name
    #     return None

    seen = []
    how_many = 0
    with open(infile) as f:
        # UserID\tTweetID\tLatitude\tLongitude\tCreatedAt\tText\tPlaceID
        for line in f:
            data = line.strip().split('\t')
            if len(data) is not 7:
                continue
            uid, tid, x, y, t, msg, place = data
            # if not id_must_be_process(int(tid)):
            #     continue
            lat, lon = float(x), float(y)
            # city = find_city(lat, lon)
            # assert city == find_town(lat, lon, tree)
            city = th.find_town(lat, lon, tree)
            lid = None
            if city is not None:
                lid = extract_url_from_msg(msg)
                stats[city] += 1
                how_many += 1
                tid, uid = int(tid), int(uid)
                t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
                t = cities.utc_to_local(city, t)
                # to have more numerical values (but lid should be a 64bit
                # unsigned integer which seems to be quite complicated in
                # mongo)
                # t = timegm(t.utctimetuple())
                # city = cities.INDEX[city]
                loc = Location('Point', [lon, lat])._asdict()
                # seen.append(CheckIn(tid, lid, uid, city, loc, t, place))
                seen.append((tid, t, place))
                if len(seen) > 2000:
                    # save_to_mongo(seen, checkins, venues_getter)
                    update_time_and_place(seen, checkins)
                    seen = []
                if how_many % 10000 == 0:
                    print('1000(0) miles more')
                    # save_var('avenues_id_new_triton', venues_getter.results)
                    # save_var('avenues_errors_triton', venues_getter.errors)

    # save_to_mongo(seen, checkins, venues_getter)
    update_time_and_place(seen, checkins)
    counts = sorted(stats.iteritems(), key=lambda x: x[1], reverse=True)
    print('\n'.join(['{}: {}'.format(city, count) for city, count in counts]))
    print('total:' + str(sum(stats.values())))
    # save_var('avenues_id_new_triton', venues_getter.results)
    # save_var('avenues_errors_triton', venues_getter.errors)
