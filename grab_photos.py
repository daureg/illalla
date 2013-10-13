#! /usr/bin/python2
# vim: set fileencoding=utf-8
import datetime
import calendar
import flickr_api
from operator import itemgetter
import re
TITLE_AND_TAGS = re.compile(r'^(?P<title>[^#]+)\s*(?P<tags>(?:#\w+\s*)*)$')


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

    return t


def simplifyPhoto(p):
    s = {}
    s['id'] = int(p.id)
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
    s['accuracy'] = p.location['accuracy']
    s['longitude'] = p.location['longitude']
    s['latitude'] = p.location['latitude']
    return s


def make_request():
    d = datetime.datetime(2013, 9, 21)
    tm = calendar.timegm(d.utctimetuple())
    SF_BL = (37.7123, -122.531)
    SF_TR = (37.7981, -122.364)
    bbox = '{},{},{},{}'.format(SF_BL[1], SF_BL[0], SF_TR[1], SF_TR[0])
    ct = 1  # photos only
    m = "photos"  # not video
    ppg = 10
    pg = 1
    ex = 'date_upload,date_taken,geo,tags'
    f = flickr_api.Photo.search(min_upload_date=tm, bbox=bbox,
                                accuracy='1-16', content_type=ct, media=m,
                                per_page=ppg, page=pg, extra=ex)
    return f


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    f = make_request()
    print(f.info.total)
