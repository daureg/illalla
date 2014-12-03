#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Functions used in twitter scrapper main code."""
import functools
from timeit import default_timer as clock
from time import sleep
import utils as u
import cities
import pytz
import ujson
import logging
from datetime import datetime, timedelta
import re
CHECKIN_URL = re.compile(r'([0-9a-f]{24})\?s=([0-9A-Za-z_-]{27})')
from collections import namedtuple
import locale
locale.setlocale(locale.LC_ALL, 'C')  # to parse date
UTC_DATE = '%a %b %d %X +0000 %Y'
FullCheckIn = namedtuple('FullCheckIn', ['id', 'lid', 'uid', 'city', 'loc',
                                         'time', 'tid', 'tuid', 'msg'])


def parse_tweet(tweet):
    """Return a CheckIn from `tweet` or None if it is not located in a valid
    city"""
    loc = u.get_nested(tweet, 'coordinates')
    city = None
    if not loc:
        # In that case, we would have to follow the link to know whether the
        # checkin falls within our cities but that's too costly so we drop it
        # (and introduce a bias toward open sharing users I guess)
        return None
    lon, lat = loc['coordinates']
    city = find_town(lat, lon, CITIES_TREE)
    if not (city and city in cities.SHORT_KEY):
        return None
    tid = u.get_nested(tweet, 'id_str')
    urls = u.get_nested(tweet, ['entities', 'urls'], [])
    # short url of the checkin that need to be expand, either using bitly API
    # or by VenueIdCrawler. Once we get the full URL, we still need to request
    # 4SQ (500 per hours) to get info.
    is_foursquare_url = lambda u: '4sq.com' in u or 'swarmapp.com' in u
    fsq_urls = [url['expanded_url'] for url in urls
                if is_foursquare_url(url['expanded_url'])]
    if not fsq_urls:
        return None
    lid = str(fsq_urls[0])
    uid = u.get_nested(tweet, ['user', 'id_str'])
    msg = u.get_nested(tweet, 'text')
    try:
        time = datetime.strptime(tweet['created_at'], UTC_DATE)
        time = cities.utc_to_local(city, time)
    except ValueError:
        print('time: {}'.format(tweet['created_at']))
        return None
    return FullCheckIn('', lid, '', city, loc, time, tid, uid, msg)


def import_json():
    """Return a json module (first trying ujson then simplejson and finally
    json from standard library)."""
    try:
        import ujson as json
    except ImportError:
        # try:
        #     import simplejson as json
        # except ImportError:
        #     import json
        # I cannot make the others two work with utf-8
        raise
    return json


def log_exception(log, default=None, reraise=False):
    """If `func` raises an exception, log it to `log`. By default, assume it's
    not critical and thus resume execution by returning `default`, except if
    `reraise` is True."""
    def actual_decorator(func):
        """Real decorator, with no argument"""
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            """Wrapper"""
            try:
                return func(*args, **kwds)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                log.exception("")
                if reraise:
                    raise
                return default
        return wrapper
    return actual_decorator


class Failures(object):
    """Keep track of Failures."""
    def __init__(self, initial_waiting_time):
        """`initial_waiting_time` is in minutes."""
        self.total_failures = 0
        self.last_failure = clock()
        self.initial_waiting_time = float(initial_waiting_time)*60.0
        self.reset()

    def reset(self):
        """Restore initial state with no recent failure."""
        self.recent_failures = 0
        self.waiting_time = self.initial_waiting_time

    def fail(self):
        """Register a new failure and return a reasonable time to wait"""
        if self.has_failed_recently():
            # Hopefully the golden ration will bring us luck next time
            self.waiting_time *= 1.618
        else:
            self.reset()
        self.total_failures += 1
        self.recent_failures += 1
        self.last_failure = clock()
        return self.waiting_time

    def has_failed_recently(self, small=3600):
        """Has it failed in the last `small` seconds?"""
        return self.total_failures > 0 and clock() - self.last_failure < small

    def do_sleep(self):
        """Indeed perform waiting."""
        sleep(self.waiting_time)


def parse_json_checkin(json, url=None):
    """Return salient info about a Foursquare checkin `json` that can be
    either JSON text or already parsed as a dictionary."""
    if not json:
        return None
    if not isinstance(json, dict):
        try:
            checkin = ujson.loads(json)
        except (TypeError, ValueError) as not_json:
            print(not_json, json, url)
            return None
    else:
        checkin = json['checkin']
    uid = u.get_nested(checkin, ['user', 'id'])
    vid = u.get_nested(checkin, ['venue', 'id'])
    time = u.get_nested(checkin, 'createdAt')
    offset = u.get_nested(checkin, 'timeZoneOffset', 0)
    if None in [uid, vid, time]:
        return None
    time = datetime.fromtimestamp(time, tz=pytz.utc)
    # by doing this, the date is no more UTC. So why not put the correct
    # timezone? Because in that case, pymongo will convert to UTC at
    # insertion. Yet I want local time, but without doing the conversion
    # when the result comes back from the DB.
    time += timedelta(minutes=offset)
    return int(uid), str(vid), time


def save_checkins_json(complete, prefix='tweets'):
    """Save `complete` as JSON in a file."""
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = '{}_{}.json'.format(prefix, now)
    msg = 'Save {} tweets in {}.'.format(len(complete), filename)
    try:
        for idx, checkin in enumerate(complete):
            fmt_time = checkin['time'].strftime('%Y-%m-%dT%H:%M:%SZ')
            complete[idx]['time'] = {'$date': fmt_time}
        with open(filename, 'w') as out:
            out.write(ujson.dumps(complete, ensure_ascii=False).replace('\/',
                                                                        '/'))
            logging.info(msg)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        msg = "Fail to save {} tweets.".format(len(complete))
        logging.exception(msg)


Point = namedtuple('Point', ['x', 'y'])
Node = namedtuple('Node', ['val', 'left', 'right'])
from numpy import median


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


def obtain_tree():
    all_cities = cities.US + cities.EU
    cities_names = [cities.short_name(c) for c in cities.NAMES]
    bboxes = [Bbox(city, name) for city, name in zip(all_cities,
                                                     cities_names)]
    return build_tree(bboxes)
CITIES_TREE = obtain_tree()
