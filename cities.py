#! /usr/bin/python2
# vim: set fileencoding=utf-8
# from more_query import bbox_to_polygon
# from json import dumps
"""A list of cities with related information: bounding box, name, local
Euclidean projection."""
from string import ascii_lowercase as alphabet
from api_keys import FLICKR_KEY as key
import LocalCartesian as lc
from datetime import datetime as dt
import pytz
import bidict
import tempfile


def photos_request(bbox):
    from calendar import timegm
    for y in range(2008,2015):
        mind = timegm(dt(y,1,1).utctimetuple())
        maxd = timegm(dt(y+1,1,1).utctimetuple())
        print('curl --silent "https://api.flickr.com/services/rest/?min_upload_date={}&max_upload_date={}&format=json&min_taken_date=1990-07-18+17%3A00%3A00&nojsoncallback=1&method=flickr.photos.search&extras=date_upload%2Cdate_taken%2Cgeo%2Ctags&bbox={}%2C{}%2C{}%2C{}&content_type=1&media=photos&per_page=1&page=1&accuracy=16&api_key={}"| jq .photos.total'.format(mind, maxd, bbox[1], bbox[0], bbox[3], bbox[2], key))


def bbox_to_polygon(bbox):
    """Return a 4 points polygon based on the bottom left and upper
    right coordinates of bbox [lat_bl, long_bl, lat_ur, long_ur]"""
    assert(len(bbox) == 4)
    lat_bl, long_bl, lat_ur, long_ur = bbox
    return [[lat_bl, long_bl], [lat_bl, long_ur],
            [lat_ur, long_ur], [lat_ur, long_bl]]


def short_name(long_name):
    """Return normalized name of city"""
    return ''.join([c.lower() for c in long_name if c.lower() in alphabet])

NYC = [40.583, -74.040, 40.883, -73.767]
WAS = [38.8515, -77.121, 38.9848, -76.902]
SAF = [37.7123, -122.531, 37.84, -122.35]
ATL = [33.657, -84.529, 33.859, -84.322]
IND = [39.632, -86.326, 39.958, -85.952]
LAN = [33.924, -118.632, 34.313, -118.172]
SEA = [47.499, -122.437, 47.735, -122.239]
HOU = [29.577, -95.686, 29.897, -95.187]
SLO = [38.535, -90.320, 38.740, -90.180]
CHI = [41.645, -87.844, 42.020, -87.520]
LON = [51.475, -0.245, 51.597, 0.034]
PAR = [48.8186, 2.255, 48.9024, 2.414]
BER = [52.389, 13.096, 52.651, 13.743]
ROM = [41.8000, 12.375, 41.9848, 12.610]
PRA = [49.9777, 14.245, 50.1703, 14.660]
MOS = [55.584, 37.353, 55.906, 37.848]
AMS = [52.3337, 4.730, 52.4175, 4.986]
HEL = [60.1463, 24.839, 60.2420, 25.0200]
STO = [59.3003, 17.996, 59.3614, 18.162]
BAR = [41.3253, 2.1004, 41.4669, 2.240]
US = [NYC, WAS, SAF, ATL, IND, LAN, SEA, HOU, SLO, CHI]
EU = [LON, PAR, BER, ROM, PRA, MOS, AMS, HEL, STO, BAR]
NAMES = ['New York', 'Washington', 'San Francisco', 'Atlanta', 'Indianapolis',
         'Los Angeles', 'Seattle', 'Houston', 'St. Louis', 'Chicago',
         'London', 'Paris', 'Berlin', 'Rome', 'Prague', 'Moscow', 'Amsterdam',
         'Helsinki', 'Stockholm', 'Barcelona']
_TIMEZONES = ['America/New_York', 'America/New_York', 'America/Los_Angeles',
              'America/New_York', 'America/Indiana/Indianapolis',
              'America/Los_Angeles', 'America/Los_Angeles', 'America/Chicago',
              'America/Chicago', 'America/Chicago', 'Europe/London',
              'Europe/Paris', 'Europe/Berlin', 'Europe/Rome', 'Europe/Prague',
              'Europe/Moscow', 'Europe/Amsterdam', 'Europe/Helsinki',
              'Europe/Stockholm', 'Europe/Madrid']
UTC_TZ = pytz.utc
SHORT_KEY = [short_name(city) for city in NAMES]
FULLNAMES = bidict.bidict(zip(SHORT_KEY, NAMES))
get_tz = lambda tz: pytz.timezone(tz).localize(dt.utcnow()).tzinfo
TZ = {city: get_tz(tz) for tz, city in zip(_TIMEZONES, SHORT_KEY)}
INDEX = {short_name(city): _id for _id, city in enumerate(NAMES)}
middle = lambda bbox: (.5*(bbox[0] + bbox[2]), (.5*(bbox[1] + bbox[3])))
GEO_TO_2D = {name: lc.LocalCartesian(*middle(city)).forward
             for name, city in zip(SHORT_KEY, US+EU)}
BBOXES = dict(zip(SHORT_KEY, [bbox_to_polygon(b) for b in US+EU]))


def euclidean_to_geo(city, coords):
    """Convert back from 2D `coords` [lat, lng] to latitude and longitude
    whithin `city` using an external program (so it's not fast)."""
    import subprocess as sp
    import os
    bounds = (US+EU)[SHORT_KEY.index(city)]
    center = list(middle(bounds))
    if isinstance(coords, lc.numpy.ndarray):
        _, fpath = tempfile.mkstemp(text=True)
        lc.numpy.savetxt(fpath, coords, fmt='%.10f %.10f 0')
        cmd = 'CartConvert -r -l {} {} 0 --input-file {}'
        output = sp.check_output(cmd.format(*(center + [fpath])), shell=True)
        os.remove(fpath)
        raw = [float(c) for line in output.split('\n')
               for c in line.split()[:2]]
        return lc.numpy.array(raw).reshape(len(coords), 2)
    cmd = 'echo {} {} 0 |CartConvert -r -l {} {} 0'
    output = sp.check_output(cmd.format(*(coords + center)), shell=True)
    return [float(c) for c in output.split()[:2]]


def utc_to_local(city, time):
    """Takes `time`, which represents a datetime in UTC (maybe implicitely) and
    return the naive datetime representing local time in `city`.
    >>> utc_to_local('newyork', dt(2009, 7, 25, 12, 45))
    datetime.datetime(2009, 7, 25, 8, 45)
    >>> utc_to_local('newyork', dt(2009, 2, 25, 12, 45))
    datetime.datetime(2009, 2, 25, 7, 45)
    """
    time = time.replace(tzinfo=UTC_TZ)
    time = TZ[city].normalize(time.astimezone(TZ[city]))
    return time.replace(tzinfo=None)


def local_to_utc(city, time):
    """Do the converse of utc_to_local.
    >>> local_to_utc('newyork', dt(2009, 7, 25, 8, 45))
    datetime.datetime(2009, 7, 25, 12, 45)
    >>> local_to_utc('newyork', dt(2009, 2, 25, 7, 45))
    datetime.datetime(2009, 2, 25, 12, 45)
    """
    time = time.replace(tzinfo=None)
    return time - TZ[city].utcoffset(time, True)

if __name__ == '__main__':
    from random import uniform, choice, randint
    from geographiclib.geodesic import Geodesic
    import doctest
    doctest.testmod()
    EARTH = Geodesic.WGS84
    city = HOU
    name = 'houston'
    place = lambda: (uniform(city[0], city[2]), uniform(city[1], city[3]))
    # photos_request(NYC)
    print(bbox_to_polygon(PAR))
    for i in range(0):
        year, month, hour = randint(2007, 2014), randint(1, 12), randint(0, 23)
        date = dt(year, month, 25, hour, 45)
        city = choice(SHORT_KEY)
        if local_to_utc(city, utc_to_local(city, date)) != date:
            print(city, date)

    for i in range(0):
        p1 = place()
        p2 = place()
        local_diff = GEO_TO_2D[name](p1) - GEO_TO_2D[name](p2)
        estimated_dst = lc.numpy.linalg.norm(local_diff[:2])
        estimated_dst_f = lc.numpy.linalg.norm(local_diff)
        real_dst = EARTH.Inverse(p1[0], p1[1], p2[0], p2[1])['s12']
        variation = 100*(estimated_dst-real_dst)/real_dst
        variation_f = 100*(estimated_dst_f-real_dst)/real_dst
        print('variation: {:.7f}% vs {:.7f}%'.format(variation, variation_f))
#     for cities in US+EU:
#         print(dumps(bbox_to_polygon(cities)))
