#! /usr/bin/python2
# vim: set fileencoding=utf-8
from collections import Counter
from persistent import load_var
import json
from random import uniform
import CommonMongo as cm


def noise():
    return uniform(0, 1e-6)


def to_css_hex(color):
    r = '#'
    for i in color[:-1]:
        c = hex(int(255*i))[2:]
        if len(c) == 2:
            r += c
        else:
            r += '0' + c
    return r


def photos_to_heat_dataset(city, precision=4, limit=300):
    photos = load_var(city)
    points = Counter([(round(p[0], precision), round(p[1], precision))
                      for p in photos])
    maxi = points.most_common(1)[0][1]
    dataset = [{'lat': p[1], 'lon': p[0], 'value': c}
               for p, c in points.most_common(limit)]
    json_dataset = json.dumps({'max': maxi, 'data': dataset})
    with open(city+'.js', 'w') as f:
        f.write('var {} = {}'.format(city, json_dataset))


def photos_to_cluster_dataset(city, limit=300):
    photos = load_var(city)
    points = [[p[0] + noise(), p[1] + noise(), 'Win!']
              for p in photos[:limit]]
    with open(city+'_cluster.js', 'w') as f:
        f.write('var {}_cluster = {}'.format(city, str(points)))


def output_checkins(city):
    """Write a JS array of all checkins in `city` with their hour."""
    checkins = cm.connect_to_db('foursquare')[0]['checkin']
    query = cm.build_query(city, venue=False, fields=['loc', 'time'])
    res = checkins.aggregate(query)['result']

    def format_checkin(checkin):
        """Extract location (plus jitter) and hour from checkin"""
        lng, lat = checkin['loc']['coordinates']
        hour = checkin['time'].hour
        return [lng + noise(), lat + noise(), hour]

    formated = [str(format_checkin(c)) for c in res]
    with open(city + '_fs.js', 'w') as output:
        output.write('var helsinki_fs = [\n')
        output.write(',\n'.join(formated))
        output.write('];')
