#! /usr/bin/python2
# vim: set fileencoding=utf-8
import datetime
from shapely.geometry import Point, mapping
import fiona


def total_seconds(td):
    if hasattr(td, 'total_seconds'):
        # python 2.7
        return td.total_seconds()
    # I don't care about microsecond
    return td.seconds + td.days * 24 * 3600


def bbox_to_polygon(bbox):
    """Return a 5 points GeoJSON polygon based on the bottom left and upper
    right coordinates of bbox [lat_bl, long_bl, lat_ur, long_ur]
    (5 because the polygon needs to be closed, see:
    https://groups.google.com/d/msg/mongodb-user/OPouYFHS_zU/cS21L0XAMkkJ )
    >>> bbox_to_polygon([37, -122, 35, -120])
    {'type': 'Polygon', 'coordinates': [[[-122, 37], [-120, 37], [-120, 35], [-122, 35], [-122, 37]]]}
    """
    assert(len(bbox) == 4)
    lat_bl, long_bl, lat_ur, long_ur = bbox
    r = {}
    r['type'] = 'Polygon'
    r['coordinates'] = [[[long_bl, lat_bl], [long_ur, lat_bl],
                         [long_ur, lat_ur], [long_bl, lat_ur],
                         [long_bl, lat_bl]]]
    return r


def inside_bbox(bbox):
    return {'$geoWithin': {'$geometry': bbox_to_polygon(bbox)}}


def get_photo_url(p, size='z', webpage=False):
    web_url = u"http://www.flickr.com/photos/{}/{}".format(p['uid'], p['_id'])
    BASE = u"http://farm{}.staticflickr.com/{}/{}_{}_{}.jpg"
    url = BASE.format(p['farm'], p['server'], p['_id'], p['secret'], size)
    if webpage:
        return url, web_url
    return url


def tag_location(collection, tag, bbox, start, end, uploaded=False):
    """Return a list of [long, lat] for each photo taken between start and end
    (or uploaded) in bbox which has tag."""
    query = {}
    field = {'loc': 1, '_id': 0}
    query['loc'] = inside_bbox(bbox)
    query['tags'] = {'$in': [tag]}
    time_field = 'upload' if uploaded else 'taken'
    query[time_field] = {'$gt': start, '$lt': end}
    cursor = collection.find(query, field)
    return map(lambda p: p['loc']['coordinates'], list(cursor))


def tag_over_time(collection, tag, bbox, start, interval):
    now = datetime.datetime.now()
    num_period = total_seconds(now - start)
    num_period = int(num_period/total_seconds(interval))
    schema = {'geometry': 'Point', 'properties': {}}
    for i in range(num_period):
        places = map(lambda p: {'geometry': mapping(Point(p[0], p[1])),
                                'properties': {}},
                     tag_location(collection, tag, bbox, start + i * interval,
                                  start + (i+1) * interval))
        print('{} - {}: {}'.format(start + i * interval,
                                   start + (i+1) * interval, len(places)))
        name = '{}_{}.shp'.format(tag, i+1)
        with fiona.collection(name, "w", "ESRI Shapefile", schema) as f:
            f.writerecords(places)
        # with open(name, 'w') as f:
        #     writer = csv.writer(f, delimiter=';')
        #     writer.writerows(places)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    tag_over_time(None, 'baseball', [37.768, -122.4, 37.778, -122.38],
                  datetime.datetime(2008, 1, 1), datetime.timedelta(days=91.3))
