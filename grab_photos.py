#! /usr/bin/python2
# vim: set fileencoding=utf-8
import datetime
import calendar
import pymongo
import json
import urllib
import urllib2
import flickr_api as flickr_api
from flickr_keys import API_KEY
import re
from time import sleep, time
from timeit import default_timer as clock
import logging
logging.basicConfig(filename='grab_photos.log',
                    level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s]: %(message)s')

TITLE_AND_TAGS = re.compile(r'^(?P<title>[^#]*)\s*(?P<tags>(?:#\w+\s*)*)$')
MACHINE_TAGS = re.compile(r'^\w+:\w+')
unique_id = set([])
BASE_URL = "http://api.flickr.com/services/rest/"
PER_PAGE = 100
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
SF_TR = (37.7981, -122.364)
NY_BL = (40.583, -74.040)
NY_TR = (40.883, -73.767)
LD_BL = (51.475, -0.245)
LD_TR = (51.597, 0.034)
VG_BL = (36.80, -78.52)
VG_TR = (38.62, -76.27)
CA_BL = (37.05, -122.21)
CA_TR = (39.59, -119.72)


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
    [u'iphoneography', u'instagramapp']
    """
    if not isinstance(s, unicode) or len(s) == 0:
        return []
    # reg = MACHINE_TAGS
    # return [t.strip() for t in s.split() if not reg.match(t)]
    return [t for t in s.split() if not ':' in t]


def photo_to_dict(p):
    start = clock()
    s = {}
    if not ('id' in p and
            'owner' in p and
            'datetaken' in p and
            'dateupload' in p and
            'tags' in p and
            'title' in p and
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
    s['tags'] = get_human_tags(p['tags'])
    if len(s['tags']) < 1:
        took = 1000*(clock() - start)
        logging.debug('map {} in {:.3f}ms (no tag)'.format(s['_id'], took))
        return None
    coord = [p['longitude'], p['latitude']]
    s['loc'] = {"type": "Point", "coordinates": coord}
    took = 1000*(clock() - start)
    logging.debug('map {} in {:.3f}ms'.format(s['_id'], took))
    return s


def save_to_mongo(photos, collection):
    global unique_id
    converted = [photo_to_dict(p) for p in photos]
    tagged = [p for p in converted if p is not None]
    total = len(tagged)
    ids = [p['_id'] for p in tagged]
    unique_id |= set(ids)
    if total > 0:
        try:
            collection.insert(tagged, continue_on_error=True)
        except pymongo.errors.DuplicateKeyError:
            # we don't really care, it means that we already have these ones
            logging.info('duplicate')
            pass
    return total


def make_request(start_time, bottom_left, upper_right, page, need_answer=False,
                 max_tries=3):
    """ Queries photos uploaded after 'start_time' in the region defined by
    'bottom_left' and 'upper_right'. If successfull, return all of them in page
    'page' along with some info. Otherwise, return None by default.  If
    'need_answer' is true, try again at most 'max_tries' times. """
    bbox = '{},{},{},{}'.format(bottom_left[1], bottom_left[0],
                                upper_right[1], upper_right[0])
    min_upload = calendar.timegm(start_time.utctimetuple())
    while max_tries > 0:
        error = False
        try:
            res, t = send_request(min_upload_date=min_upload,
                                  min_taken_date='1990-07-18 17:00:00',
                                  bbox=bbox, accuracy='16',
                                  content_type=1,  # photos only
                                  media="photos",  # not video
                                  per_page=50, page=page,
                                  extras='date_upload,date_taken,geo,tags')
        except flickr_api.FlickrError as e:
            logging.warn('Error getting page {}: {}'.format(page, e))
            error = True

        if not error and len(res) > 0:
            return res, t
        if need_answer:
            max_tries -= 1
            logging.info('sleeping on page {}'.format(page))
            sleep(8)
        else:
            return None, 0
    return None


if __name__ == '__main__':
    # import doctest
    # doctest.testmod()
    import sys
    first_page = 1 if len(sys.argv) < 2 else int(sys.argv[1])

    START_OF_REQUESTS = time()
    logging.info('initial request')
    start_time = datetime.datetime(2008, 1, 1)
    # f, t = make_request(start_time, SF_BL, SF_TR, 1, need_answer=True)
    # if f is None:
    #     logging.critical('cannot pass the first request')
    #     print('cannot pass the first request')
    #     sys.exit()

    # num_pages = t
    # print(num_pages)
    num_pages = 50
    client = pymongo.MongoClient('localhost', 27017)
    db = client['flickr']
    photos = db['photos']
    photos.ensure_index([('loc', pymongo.GEOSPHERE),
                         ('tags', pymongo.ASCENDING),
                         ('uid', pymongo.ASCENDING)])
    failed_page = []
    total = 0
    for page in range(first_page, num_pages+1):
        start = clock()
        res, _ = make_request(start_time, SF_BL, SF_TR, page)
        if res is None:
            failed_page.append(page)
        else:
            took = ' ({:.4f}s)'.format(clock() - start)
            logging.info('Get result for page {}{}'.format(page, took))
            saved = save_to_mongo(res, photos)
            took = ' ({:.4f}s)'.format(clock() - start)
            page_desc = 'page {}, {} photos {}'.format(page, saved, took)
            logging.info('successfully insert ' + page_desc)
            total += saved
            sleep(1)
    for page in failed_page:
        start = clock()
        res, _ = make_request(start_time, SF_BL, SF_TR, page, need_answer=True)
        if res is None:
            took = ' ({:.4f}s)'.format(clock() - start)
            logging.warn('Failed to get page {}{}'.format(page, took))
        else:
            saved = save_to_mongo(res, photos)
            took = ' ({:.4f}s)'.format(clock() - start)
            page_desc = 'page {}, {} photos {}'.format(page, saved, took)
            logging.info('Finally get ' + page_desc)
            total += saved
            sleep(5)

    logging.info('Saved a total of {} photos.'.format(total))
    uniq = len(unique_id)
    logging.info('or {} photos ({}% duplicate).'.format(uniq,
                                                        100*(1-uniq)/total))
    logging.info('made {} requests.'.format(TOTAL_REQ))
