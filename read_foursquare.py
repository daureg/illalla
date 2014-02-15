#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Read ICWSM 11 Foursquare dataset and keep only check in occuring in
specified cities.
In the following, x refers to the latitude and y to the longitude"""
import cities
from collections import namedtuple
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
    split_coord = depth % 2
    split_val = median([b.bottom[split_coord] for b in bboxes])
    left, right = [], []
    for b in bboxes:
        if b.center[split_coord] > split_val:
            right.append(b)
        else:
            left.append(b)
    return Node(split_val,
                build_tree(left, depth+1), build_tree(right, depth+1))


def find_town(x, y, tree, depth=0):
    if isinstance(tree, list):
        # print('for {}, {}: reach leaf: {}'.format(x, y, tree))
        for city in tree:
            # print('is ({}, {}) contained in {}'.format(x, y, city))
            if city.contains(x, y):
                # print('indeed the point is in ' + str(city))
                return city.name
        return None
    point_val = x if depth % 2 == 0 else y
    if point_val > tree.val:
        return find_town(x, y, tree.right, depth+1)
    else:
        return find_town(x, y, tree.left, depth+1)


def center(p1, p2):
    assert isinstance(p1, Point) and isinstance(p2, Point)
    return Point(.5*(p1.x + p2.x), .5*(p1.y + p2.y))


class Bbox():
    bottom = None
    top = None
    center = None
    name = None

    def __init__(self, bbox, name):
        self.bottom = Point(*bbox[:2])
        self.top = Point(*bbox[2:])
        self.center = center(self.bottom, self.top)
        self.name = name

    def contains(self, x, y):
        return self.bottom.x <= x <= self.top.x and\
            self.bottom.y <= y <= self.top.y

    def __repr__(self):
        return '{}: {:.2f}, {:.2f}'.format(self.name, self.center.x,
                                           self.center.y)

if __name__ == '__main__':
    # import doctest
    # doctest.testmod()
    import sys
    infile = 'verysmall' if len(sys.argv) < 2 else sys.argv[1]
    all_cities = cities.US + cities.EU
    cities_names = [cities.short_name(c) for c in cities.NAMES]
    bboxes = [Bbox(city, name) for city, name in zip(all_cities,
                                                     cities_names)]
    tree = build_tree(bboxes)

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
            city = find_city(lat, lon)
            assert city == find_town(lat, lon, tree)
            if city is not None:
                tid, lid, uid = int(tid), int(lid, 16), int(uid)
                t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
                # to have numerical values
                # t = timegm(t.utctimetuple())
                # city = cities.INDEX[city]
                res.append(CheckIn(tid, lid, uid, city, lat, lon, t))
