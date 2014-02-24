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
from numpy import median
from bisect import bisect_left
# from calendar import timegm

Point = namedtuple('Point', ['x', 'y'])
Location = namedtuple('Location', ['type', 'coordinates'])
CheckIn = namedtuple('CheckIn',
                     ['tid', 'lid', 'uid', 'city', 'loc', 'time'])
Node = namedtuple('Node', ['val', 'left', 'right'])
BLACKLIST = ['fst.je', 'gowal.la', 'picplz.com', 'wal.la', 'flic.kr',
             'myloc.me', 'wp.me', 'yfrog.com', 'j.mp', 'bkite.com', 'untpd.it']
MISSING_ID = []


def build_tree(bboxes, depth=0, max_depth=2):
    if depth >= max_depth:
        return bboxes
    split_val = median([b.bottom[1] for b in bboxes])
    left, right = [], []
    for b in bboxes:
        if b.bottom[1] > split_val:
            right.append(b)
        else:
            left.append(b)
    return Node(split_val,
                build_tree(left, depth+1), build_tree(right, depth+1))


def find_town(x, y, tree, depth=0):
    if isinstance(tree, list):
        for city in tree:
            if city.contains(x, y):
                return city.name
        return None
    if y > tree.val:
        return find_town(x, y, tree.right, depth+1)
    else:
        return find_town(x, y, tree.left, depth+1)


class Bbox():
    bottom = None
    top = None
    center = None
    name = None

    def __init__(self, bbox, name):
        self.bottom = Point(*bbox[:2])
        self.top = Point(*bbox[2:])
        self.name = name

    def contains(self, x, y):
        return self.bottom.x <= x <= self.top.x and\
            self.bottom.y <= y <= self.top.y

    def __repr__(self):
        return '{}: {:.2f}, {:.2f}'.format(self.name, self.bottom.x,
                                           self.bottom.y)


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
    return
    converted = []
    for i, c in enumerate(documents):
        converted.append(convert_checkin_for_mongo(c))
        converted[-1]['lid'] = ids[i]
    try:
        destination.insert(converted, continue_on_error=True)
    except pymongo.errors.DuplicateKeyError:
        # it's not my dataset so there is no much I can do
        pass


def id_must_be_process(_id):
    i = bisect_left(MISSING_ID, _id)
    return i != len(MISSING_ID) and MISSING_ID[i] == _id


def obtain_tree():
    all_cities = cities.US + cities.EU
    cities_names = [cities.short_name(c) for c in cities.NAMES]
    bboxes = [Bbox(city, name) for city, name in zip(all_cities,
                                                     cities_names)]
    return build_tree(bboxes)

if __name__ == '__main__':
    # import doctest
    # doctest.testmod()
    from persistent import save_var, load_var
    MISSING_ID = load_var('tid_diff')

    previously = load_var('t_venues_id_new')
    venues_getter = VenueIdCrawler(previously, True)

    checkins = None
    # client = pymongo.MongoClient('localhost', 27017)
    # db = client['foursquare']
    # checkins = db['checkin']
    # checkins.ensure_index([('loc', pymongo.GEOSPHERE),
    #                        ('lid', pymongo.ASCENDING),
    #                        ('city', pymongo.ASCENDING),
    #                        ('time', pymongo.ASCENDING)])
    import sys
    infile = 'verysmall' if len(sys.argv) < 2 else sys.argv[1]
    tree = obtain_tree()
    stats = defaultdict(lambda: 0)

    # def find_city(x, y):
    #     for city in bboxes:
    #         if city.contains(x, y):
    #             return city.name
    #     return None

    seen = []
    with open(infile) as f:
        # UserID\tTweetID\tLatitude\tLongitude\tCreatedAt\tText\tPlaceID
        for line in f:
            data = line.strip().split('\t')
            if len(data) is not 7:
                continue
            uid, tid, x, y, t, msg, _ = data
            # if not id_must_be_process(int(tid)):
            #     continue
            lat, lon = float(x), float(y)
            # city = find_city(lat, lon)
            # assert city == find_town(lat, lon, tree)
            city = find_town(lat, lon, tree)
            lid = None
            if city is not None:
                last_word = msg.split()[-1]
                if last_word.startswith('htt') and len(last_word) < 24:
                    host = urlparse.urlparse(last_word).netloc
                    if host in BLACKLIST:
                        last_word = None
                    lid = last_word
                stats[city] += 1
                tid, uid = int(tid), int(uid)
                t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
                # to have more numerical values (but lid should be a 64bit
                # unsigned integer which seems to be quite complicated in
                # mongo)
                # t = timegm(t.utctimetuple())
                # city = cities.INDEX[city]
                loc = Location('Point', [lon, lat])._asdict()
                seen.append(CheckIn(tid, lid, uid, city, loc, t))
                if len(seen) > 1000:
                    save_to_mongo(seen, checkins, venues_getter)
                    seen = []
                save_var('venues_id_new', venues_getter.results)

    save_to_mongo(seen, checkins, venues_getter)
    counts = sorted(stats.iteritems(), key=lambda x: x[1], reverse=True)
    print('\n'.join(['{}: {}'.format(city, count) for city, count in counts]))
    print('total:' + str(sum(stats.values())))
    save_var('venues_id_new', venues_getter.results)
    save_var('venues_errors', venues_getter.errors)
    save_var('url_todo', venues_getter.todo)
