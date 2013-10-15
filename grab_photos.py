#! /usr/bin/python2
# vim: set fileencoding=utf-8
import datetime
import calendar
import flickr_api
from operator import itemgetter
import re
from time import clock
import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG)

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


def photo_to_dict(p):
    start = clock()
    s = {}
    s['_id'] = int(p.id)
    s['uid'] = p.owner.id
    s['taken'] = datetime.datetime.strptime(p.taken, '%Y-%m-%d %H:%M:%S')
    # The 'posted' date represents the time at which the photo was uploaded to
    # Flickr. It's always passed around as a unix timestamp (seconds since Jan
    # 1st 1970 GMT). It's up to the application provider to format them using
    # the relevant viewer's timezone.
    s['upload'] = datetime.datetime.fromtimestamp(p.posted)
    title, tags = parse_title(p.title)
    s['title'] = title
    s['tags'] = map(itemgetter('text'),
                    filter(lambda x: x.machine_tag == 0, p.tags)) + tags
    if len(s['tags']) < 1:
        logging.info('map {} in {:.3f}s'.format(s['_id'], clock() - start))
        return None
    coord = [p.location['longitude'], p.location['latitude']]
    s['loc'] = {"type": "Point", "coordinates": coord}
    logging.info('map {} in {:.3f}s'.format(s['_id'], clock() - start))
    return s


def save_to_mongo(photos, collection):
    collection.insert([photo_to_dict(p) for p in photos if p is not None])


def make_request(start_time, bottom_left, upper_right):
    tm = calendar.timegm(start_time.utctimetuple())
    bbox = '{},{},{},{}'.format(bottom_left[1], bottom_left[0],
                                upper_right[1], upper_right[0])
    ct = 1  # photos only
    m = "photos"  # not video
    ppg = 10
    pg = 16
    ex = 'date_upload,date_taken,geo,tags'
    f = flickr_api.Photo.search(min_upload_date=tm, bbox=bbox,
                                accuracy='16', content_type=ct, media=m,
                                per_page=ppg, page=pg, extra=ex)
    return f

if __name__ == '__main__':
    # import doctest
    # doctest.testmod()
    f = make_request(datetime.datetime(2008, 8, 1), SF_BL, SF_TR)
    print(f.info.total)
    # p = photo_to_dict(f[0])
    # import cPickle
    # with open('test_photo', 'wb') as f:
    #     pkl = cPickle.Pickler(f, 2)
    #     pkl.dump(p)
    # with open('test_photo', 'rb') as f:
    #     pkl = cPickle.Unpickler(f)
    #     p = pkl.load()
    # print(p)
    import pymongo
    client = pymongo.MongoClient('localhost', 27017)
    db = client['flickr']
    photos = db['photos']
    photos.ensure_index([('loc', pymongo.GEOSPHERE),
                         ('tags', pymongo.ASCENDING),
                         ('uid', pymongo.ASCENDING)])
    save_to_mongo(f, photos)
