#! /usr/bin/python2
# vim: set fileencoding=utf-8
from collections import Counter
from persistent import load_var
import json


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
