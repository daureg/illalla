#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Read ICWSM 11 Foursquare dataset and keep only check in occuring in
specified cities.
In the following, x refers to the latitude and y to the longitude"""
import cities
from collections import namedtuple, defaultdict
from datetime import datetime
from numpy import median
# from calendar import timegm

Point = namedtuple('Point', ['x', 'y'])
CheckIn = namedtuple('CheckIn',
                     ['tid', 'lid', 'uid', 'city', 'lat', 'lon', 't'])
Node = namedtuple('Node', ['val', 'left', 'right'])


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

if __name__ == '__main__':
    # import doctest
    # doctest.testmod()
    import sys
    infile = 'medium' if len(sys.argv) < 2 else sys.argv[1]
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

    res = []
    with open(infile) as f:
        # UserID\tTweetID\tLatitude\tLongitude\tCreatedAt\tText\tPlaceID
        for line in f:
            data = line.strip().split('\t')
            if len(data) is not 7:
                continue
            uid, tid, x, y, t, _, lid = data
            lat, lon = float(x), float(y)
            # city = find_city(lat, lon)
            # assert city == find_town(lat, lon, tree)
            city = find_town(lat, lon, tree)
            if city is not None:
                stats[city] += 1
                tid, lid, uid = int(tid), int(lid, 16), int(uid)
                t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
                # to have numerical values
                # t = timegm(t.utctimetuple())
                # city = cities.INDEX[city]
                res.append(CheckIn(tid, lid, uid, city, lat, lon, t))
    counts = sorted(stats.iteritems(), key=lambda x: x[1], reverse=True)
    print('\n'.join(['{}: {}'.format(city, count) for city, count in counts]))
