#! /usr/bin/python2
# vim: set fileencoding=utf-8
import datetime
import calendar
import flickr_api as flickr_api
from operator import itemgetter
import re
from time import sleep
from timeit import default_timer as clock
import logging
logging.basicConfig(filename='grab_photos.log',
                    level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s]: %(message)s')

TITLE_AND_TAGS = re.compile(r'^(?P<title>[^#]+)\s*(?P<tags>(?:#\w+\s*)*)$')
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


def parse_title(t):
    """ Separate title from terminal hashtags
    >>> parse_title('Carnitas Crispy Taco with guac #foodporn #tacosrule')
    ('Carnitas Crispy Taco with guac', ['foodporn', 'tacosrule'])
    >>> parse_title('Carnitas Crispy Taco with guac')
    ('Carnitas Crispy Taco with guac', [])
    """
    m = TITLE_AND_TAGS.match(t)
    if m is not None:
        title = m.group('title').strip()
        tags = m.group('tags').replace('#', '').split()
        return title, tags

    return t, []


def human_tag(t):
    return (isinstance(t, flickr_api.Tag) and
            hasattr(t, 'machine') and
            hasattr(t, 'text') and
            t.machine == 0)


def photo_to_dict(p):
    start = clock()
    s = {}
    if not (hasattr(p, 'id') and
            hasattr(p, 'owner') and
            hasattr(p.owner, 'id') and
            hasattr(p, 'taken') and
            hasattr(p, 'posted') and
            hasattr(p, 'tags') and
            isinstance(p.tags, list) and
            hasattr(p, 'title') and
            hasattr(p, 'location')):
        took = 1000*(clock() - start)
        logging.debug('map {} in {:.3f}ms (missing)'.format(p.id, took))
        return None
    try:
        s['_id'] = int(p.id)
    except ValueError:
        logging.info(str(p.id) + 'is not a valid id')
        return None
    logging.debug(p.id)
    s['uid'] = p.owner.id
    try:
        s['taken'] = datetime.datetime.strptime(p.taken, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return None
    # The 'posted' date represents the time at which the photo was uploaded to
    # Flickr. It's always passed around as a unix timestamp (seconds since Jan
    # 1st 1970 GMT). It's up to the application provider to format them using
    # the relevant viewer's timezone.
    try:
        s['upload'] = datetime.datetime.fromtimestamp(p.posted)
    except ValueError:
        return None
    title, tags = parse_title(p.title)
    s['title'] = title
    s['tags'] = map(itemgetter('text'), filter(human_tag, p.tags)) + tags
    if len(s['tags']) < 1:
        took = 1000*(clock() - start)
        logging.debug('map {} in {:.3f}ms (no tag)'.format(s['_id'], took))
        return None
    coord = [p.location['longitude'], p.location['latitude']]
    s['loc'] = {"type": "Point", "coordinates": coord}
    took = 1000*(clock() - start)
    logging.debug('map {} in {:.3f}ms'.format(s['_id'], took))
    return s


def save_to_mongo(photos, collection):
    converted = [photo_to_dict(p) for p in photos]
    tagged = [p for p in converted if p is not None]
    total = len(tagged)
    if total > 0:
        try:
            collection.insert(tagged, continue_on_error=True)
        except pymongo.errors.DuplicateKeyError:
            # we don't really care, it means that we already have these ones
            pass
    return total


def make_request(start_time, bottom_left, upper_right, page, need_answer=False,
                 max_tries=5):
    """ Queries photos uploaded after 'start_time' in the region defined by
    'bottom_left' and 'upper_right'. If successfull, return all of them in page
    'page' along with some info. Otherwise, return None by default.  If
    'need_answer' is true, try again at most 'max_tries' times. """
    bbox = '{},{},{},{}'.format(bottom_left[1], bottom_left[0],
                                upper_right[1], upper_right[0])
    min_upload = calendar.timegm(start_time.utctimetuple())
    error = False
    info = 'date_upload,date_taken,geo,tags'
    while max_tries > 0:
        try:
            res = flickr_api.Photo.search(min_upload_date=min_upload,
                                          bbox=bbox, accuracy='16',
                                          content_type=1,  # photos only
                                          media="photos",  # not video
                                          per_page=250, page=page, extra=info)
        except flickr_api.FlickrError as e:
            logging.warn('Error getting page {}: {}'.format(page, e))
            error = True
        except KeyError:
            logging.warn('Error getting page {}: probably empty'.format(page))
            error = True

        if not error:
            return res
        if need_answer:
            max_tries -= 1
            logging.info('sleeping on page {}'.format(page))
            sleep(2)
        else:
            return None
    return None


if __name__ == '__main__':
    # import doctest
    # doctest.testmod()
    import sys
    first_page = 1 if len(sys.argv) < 2 else int(sys.argv[1])

    logging.info('initial request')
    start_time = datetime.datetime(2008, 8, 1)
    f = make_request(start_time, SF_BL, SF_TR, 1, need_answer=True)
    if f is None:
        logging.CRITICAL('cannot pass the first request')
        print('cannot pass the first request')
        sys.exit()

    num_pages = f.info.pages
    print(num_pages)
    num_pages = 12
    import pymongo
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
        res = make_request(start_time, SF_BL, SF_TR, page)
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
        res = make_request(start_time, SF_BL, SF_TR, page, need_answer=True)
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

    logging.info('Saved a total of {} photos.'.format(total))
