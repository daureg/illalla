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
# from calendar import timegm

Point = namedtuple('Point', ['x', 'y'])
Location = namedtuple('Location', ['type', 'coordinates'])
CheckIn = namedtuple('CheckIn',
                     ['tid', 'lid', 'uid', 'city', 'loc', 'time'])
Node = namedtuple('Node', ['val', 'left', 'right'])
BLACKLIST = ['flic.kr', 'yfrog.com', 'yfrog.us', 'fst.je', 'gowal.la',
             'myloc.me', 'bkite.com', 'tl.gd', 'j.mp', 'picplz.com', 'bit.ly',
             'wp.me']


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
    converted = []
    for i, c in enumerate(documents):
        converted.append(convert_checkin_for_mongo(c))
        converted[-1]['lid'] = ids[i]
    try:
        destination.insert(converted, continue_on_error=True)
    except pymongo.errors.DuplicateKeyError:
        # it's not my dataset so there is no much I can do
        pass


if __name__ == '__main__':
    # import doctest
    # doctest.testmod()
    from persistent import save_var

    venues_getter = VenueIdCrawler()
    client = pymongo.MongoClient('localhost', 27017)
    db = client['foursquare']
    checkins = db['checkin']
    checkins.ensure_index([('loc', pymongo.GEOSPHERE),
                           ('lid', pymongo.ASCENDING),
                           ('city', pymongo.ASCENDING),
                           ('time', pymongo.ASCENDING)])
    import sys
    infile = 'verysmall' if len(sys.argv) < 2 else sys.argv[1]
    all_cities = cities.US + cities.EU
    cities_names = [cities.short_name(c) for c in cities.NAMES]
    bboxes = [Bbox(city, name) for city, name in zip(all_cities,
                                                     cities_names)]
    tree = build_tree(bboxes)
    stats = defaultdict(lambda: 0)

    def find_city(x, y):
        for city in bboxes:
            if city.contains(x, y):
                return city.name
        return None

    seen = []
    with open(infile) as f:
        # UserID\tTweetID\tLatitude\tLongitude\tCreatedAt\tText\tPlaceID
        for line in f:
            data = line.strip().split('\t')
            if len(data) is not 7:
                continue
            uid, tid, x, y, t, msg, _ = data
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
            if lid is not None:
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
                if len(seen) > 5000:
                    save_to_mongo(seen, checkins, venues_getter)
                    seen = []

    save_to_mongo(seen, checkins, venues_getter)
    counts = sorted(stats.iteritems(), key=lambda x: x[1], reverse=True)
    print('\n'.join(['{}: {}'.format(city, count) for city, count in counts]))
    save_var('venues_id', venues_getter.results)
