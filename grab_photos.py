#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Try its best to retrieve a list of all photos taken in a given city and
insert them with additional information in a mongo database."""
import datetime
import calendar
import CommonMongo as cm
import json
import urllib
import urllib2
import flickr_api as flickr_api
from api_keys import FLICKR_KEY as API_KEY
import re
from time import sleep, time
from timeit import default_timer as clock
from httplib import BadStatusLine
import cities
import logging
import os
import arguments
import sys
now = datetime.datetime.now().strftime('%Y%m%d_%H%M')
HINT = now if len(sys.argv) < 2 else sys.argv[1].strip()
LOG_FILE = 'photos_{}.log'.format(HINT)
TMPDIR = '/tmp' if 'TMPDIR' not in os.environ else os.environ['TMPDIR']
logging.basicConfig(filename=os.path.join(TMPDIR, LOG_FILE),
                    level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s')

TITLE_AND_TAGS = re.compile(r'^(?P<title>[^#]*)\s*(?P<tags>(?:#\w+\s*)*)$')
BASE_URL = "http://api.flickr.com/services/rest/"
PER_PAGE = 225
# According to https://secure.flickr.com/services/developer/api/, one api key
# can only make 3600 request per hour so we need to keep track of our usage to
# stay under the limit.
# TODO: move this logic to a RequestSupervisor class.
# NOTE: Actually, it's probably useless since on average, request take more
# than one second to complete
CURRENT_REQ = 0
TOTAL_REQ = 0
START_OF_REQUESTS = 0
REQUEST_INTERVAL = 3600  # in second
MAX_REQUEST = 3600

SF_BL = (37.7123, -122.531)
SF_TR = (37.84, -122.35)
NY_BL = (40.583, -74.040)
NY_TR = (40.883, -73.767)
LD_BL = (51.475, -0.245)
LD_TR = (51.597, 0.034)
VG_BL = (36.80, -78.52)
VG_TR = (38.62, -76.27)
CA_BL = (37.05, -122.21)
CA_TR = (39.59, -119.72)
US_BL = (26, -124.1)
US_TR = (48.6, -66.6)
NANTES_BL = [47.195, -1.61]
NANTES_UR = [47.27, -1.5]


def send_request(**args):
    global CURRENT_REQ, START_OF_REQUESTS, TOTAL_REQ
    if CURRENT_REQ > MAX_REQUEST:
        now = time()
        next_time = START_OF_REQUESTS + REQUEST_INTERVAL
        if now < next_time:
            pause = next_time - now + 2
            logging.info("made {} request in {}s: sleeping for {}s{}".format(
                CURRENT_REQ, now - START_OF_REQUESTS, pause,
                " (but then I come back well rested, raring to go!)"))
            sleep(pause)
        START_OF_REQUESTS = now
        TOTAL_REQ += CURRENT_REQ
        CURRENT_REQ = 0
    else:
        TOTAL_REQ = CURRENT_REQ+1

    args['method'] = 'flickr.photos.search'
    args['format'] = 'json'
    args['api_key'] = API_KEY
    args['nojsoncallback'] = 1
    req = urllib2.Request(BASE_URL, urllib.urlencode(args))
    try:
        r = json.loads(urllib2.urlopen(req).read())
        CURRENT_REQ += 1
        return r['photos']['photo'], r['photos']['total']
    except urllib2.HTTPError as e:
        raise flickr_api.FlickrError(e.read().split('&')[0])
    except BadStatusLine:
        raise flickr_api.FlickrError('BadStatusLine')
    except:
        raise


def parse_title(t):
    """ Separate title from terminal hashtags
    >>> parse_title('Carnitas Crispy Taco with guac #foodporn #tacosrule')
    ('Carnitas Crispy Taco with guac', ['foodporn', 'tacosrule'])
    >>> parse_title('#foodporn #tacosrule')
    ('', ['foodporn', 'tacosrule'])
    >>> parse_title('Carnitas Crispy Taco with guac')
    ('Carnitas Crispy Taco with guac', [])
    """
    if not '#' in t:
        return t, []
    m = TITLE_AND_TAGS.match(t)
    if m is not None:
        title = m.group('title').strip()
        tags = m.group('tags').replace('#', '').split()
        return title, tags

    return t, []


def get_human_tags(s):
    """
    >>> get_human_tags(u'iphoneography instagramapp uploaded:by=instagram')
    ([u'iphoneography', u'instagramapp'], None)
    >>> get_human_tags(u'square {foursquare}:{venue}=4bd1db7f9854d13a8260fa4d')
    ([u'square'], u'4bd1db7f9854d13a8260fa4d')
    >>> get_human_tags(u'square foursquare:venue=4bd1db7f9854d13a8260fa4d')
    ([u'square'], u'4bd1db7f9854d13a8260fa4d')
    """
    if not isinstance(s, unicode) or len(s) == 0:
        return [], None
    tags = []
    venue = None
    for t in s.split():
        if not ':' in t:
            tags.append(t)
        else:
            if venue is None and 'foursquare' in t and 'venue' in t:
                venue = t.split('=')[-1]
    return tags, venue


def photo_to_dict(p):
    global HINT
    start = clock()
    s = {}
    if not ('id' in p and
            'owner' in p and
            'datetaken' in p and
            'dateupload' in p and
            'tags' in p and
            'title' in p and
            'farm' in p and
            'secret' in p and
            'server' in p and
            'longitude' in p and
            'latitude' in p):
        took = 1000*(clock() - start)
        logging.debug('map {} in {:.3f}ms (missing)'.format(p['id'], took))
        return None
    try:
        s['_id'] = int(p['id'])
    except ValueError:
        logging.info(str(p['id']) + 'is not a valid id')
        return None
    logging.debug(p['id'])
    s['uid'] = p['owner']
    try:
        s['taken'] = datetime.datetime.strptime(p['datetaken'],
                                                '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return None
    # The 'posted' date represents the time at which the photo was uploaded to
    # Flickr. It's always passed around as a unix timestamp (seconds since Jan
    # 1st 1970 GMT). It's up to the application provider to format them using
    # the relevant viewer's timezone.
    try:
        s['upload'] = datetime.datetime.fromtimestamp(float(p['dateupload']))
    except ValueError:
        return None
    title, tags = parse_title(p['title'])
    s['title'] = title
    explicit_tag, venue = get_human_tags(p['tags'])
    s['tags'] = explicit_tag + tags
    s['venue'] = venue
    if len(s['tags']) < 1:
        took = 1000*(clock() - start)
        logging.debug('map {} in {:.3f}ms (no tag)'.format(s['_id'], took))
        return None
    # pymongo.errors.OperationFailure: Can't extract geo keys from object,
    # malformed geometry?:
    # {type: "Point", coordinates: [ "0.000000", 51.486554 ] }
    try:
        lng, lat = float(p['longitude']), float(p['latitude'])
    except ValueError:
        return None
    s['loc'] = {"type": "Point", "coordinates": [lng, lat]}
    s['farm'] = p['farm']
    s['server'] = p['server']
    s['secret'] = p['secret']
    s['hint'] = HINT
    took = 1000*(clock() - start)
    logging.debug('map {} in {:.3f}ms'.format(s['_id'], took))
    return s


def higher_request(start_time, bbox, db, level=0):
    """ Try to insert all photos in this region into db by potentially making
    recursing call, eventually to lower_request when the region accounts for
    less than 4000 photos. """
    if level > 20:
        logging.warn("Going too deep with {}.".format(bbox))
        return 0

    _, total = make_request(start_time, bbox, 1, need_answer=True,
                            max_tries=10)
    if total > 4000:
        photos = 0
        start = clock()
        quads = split_bbox(bbox)
        for q in quads:
            photos += higher_request(start_time, q, db, level+1)
        logging.info('Finish {}: {} photos in {}s'.format(bbox, photos,
                                                          clock()-start))
        return photos
    if total > 5:
        return lower_request(start_time, bbox, db, total/PER_PAGE + 1)
    logging.warn('Cannot get any photos in {}.'.format(bbox))
    return 0


def lower_request(start_time, bbox, db, num_pages):
    failed_page = []
    total = 0
    hstart = clock()
    for page in range(1, num_pages+1):
        start = clock()
        res, _ = make_request(start_time, bbox, page)
        if res is None:
            failed_page.append(page)
        else:
            took = ' ({:.4f}s)'.format(clock() - start)
            logging.info('Get result for page {}{}'.format(page, took))
            saved = save_to_mongo(res, db)
            took = ' ({:.4f}s)'.format(clock() - start)
            page_desc = 'page {}, {} photos {}'.format(page, saved, took)
            logging.info('successfully insert ' + page_desc)
            total += saved
            sleep(1)
    for page in failed_page:
        start = clock()
        res, _ = make_request(start_time, bbox, page, need_answer=True)
        if res is None:
            took = ' ({:.4f}s)'.format(clock() - start)
            logging.warn('Failed to get page {}{}'.format(page, took))
        else:
            saved = save_to_mongo(res, photos)
            took = ' ({:.4f}s)'.format(clock() - start)
            page_desc = 'page {}, {} photos {}'.format(page, saved, took)
            logging.info('Finally get ' + page_desc)
            total += saved
            sleep(1)
    logging.info('Finish {}: {} photos in {}s'.format(bbox, total,
                                                      clock()-hstart))
    return total


def save_to_mongo(photos, collection):
    global unique_id
    converted = [photo_to_dict(p) for p in photos]
    tagged = [p for p in converted if p is not None]
    total = len(tagged)
    if total > 0:
        try:
            collection.insert(tagged, continue_on_error=True)
        except cm.pymongo.errors.DuplicateKeyError:
            # we don't really care, it means that we already have these ones
            logging.info('duplicate')
            pass
    return total


def split_bbox(bbox):
    """
    >>> split_bbox(((0, 0), (20, 22)))
    [((0, 0), (10, 11)), ((0, 11), (10, 22)), ((10, 0), (20, 11)), ((10, 11), (20, 22))]
    """
    bottom_left = bbox[0]
    upper_right = bbox[1]
    bl_increment = (upper_right[1] - bottom_left[1])/2
    ur_increment = (upper_right[0] - bottom_left[0])/2
    p1 = bottom_left
    p2 = (bottom_left[0], bottom_left[1]+bl_increment)
    p3 = (bottom_left[0]+1*ur_increment, bottom_left[1]+0*bl_increment)
    p4 = (bottom_left[0]+1*ur_increment, bottom_left[1]+1*bl_increment)
    p5 = (bottom_left[0]+1*ur_increment, bottom_left[1]+2*bl_increment)
    p6 = (bottom_left[0]+2*ur_increment, bottom_left[1]+1*bl_increment)
    p7 = (bottom_left[0]+2*ur_increment, bottom_left[1]+2*bl_increment)
    return [(p1, p4), (p2, p5), (p3, p6), (p4, p7)]


def make_request(start_time, bbox, page, need_answer=False, max_tries=3):
    """ Queries photos uploaded after 'start_time' in the region defined by
    'bbox'. If successful, return all of them in page 'page' along with some
    info. Otherwise, return None by default.  If 'need_answer' is true, try
    again at most 'max_tries' times. """
    bbox = '{:.9f},{:.9f},{:.9f},{:.9f}'.format(bbox[0][1], bbox[0][0],
                                                bbox[1][1], bbox[1][0])
    min_upload = calendar.timegm(start_time.utctimetuple())
    # max_upload = calendar.timegm(datetime.datetime.now().utctimetuple())
    max_upload = calendar.timegm(datetime.datetime(2011,1,1).utctimetuple())
    while max_tries > 0:
        error = False
        try:
            res, t = send_request(min_upload_date=min_upload,
                                  max_upload_date=max_upload,
                                  min_taken_date='1990-07-18 17:00:00',
                                  bbox=bbox, accuracy='16',
                                  content_type=1,  # photos only
                                  media="photos",  # not video
                                  per_page=PER_PAGE, page=page,
                                  extras='date_upload,date_taken,geo,tags')
        except flickr_api.FlickrError as e:
            logging.warn('Error getting page {}: {}'.format(page, e))
            error = True
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            error = True

        if not error and len(res) > 0:
            return res, int(t)
        if need_answer:
            max_tries -= 1
            logging.info('insisting on page {}'.format(page))
            sleep(5)
        else:
            return None, 0

    logging.warn('Error getting page {}: too much tries'.format(page))
    return None, 0


if __name__ == '__main__':
    START_OF_REQUESTS = time()
    logging.info('initial request')

    args = arguments.city_parser().parse_args()
    photos = cm.connect_to_db('world', args.host, args.port)[0]['photos']
    photos.ensure_index([('loc', cm.pymongo.GEOSPHERE),
                         ('tags', cm.pymongo.ASCENDING),
                         ('uid', cm.pymongo.ASCENDING)])
    city = args.city
    CITY = (cities.US + cities.EU)[cities.INDEX[city]]
    HINT = city
    bbox = (CITY[:2], CITY[2:])
    start_time = datetime.datetime(2008, 1, 1)
    total = higher_request(start_time, bbox, photos)

    logging.info('Saved a total of {} photos.'.format(total))
    logging.info('made {} requests.'.format(TOTAL_REQ))
