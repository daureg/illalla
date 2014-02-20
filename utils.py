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


def photos_to_heat_dataset(city):
    photos = load_var(city)
    points = Counter([(p[0], p[1]) for p in photos])
    maxi = points.most_common(1)[0][1]
    dataset = [{'x': p[0], 'y': p[1], 'count': c}
               for p, c in points.most_common(200)]
    json_dataset = json.dumps({'max': maxi, 'data': dataset})
    with open(city+'.js', 'w') as f:
        f.write('var {} = {}'.format(city, json_dataset))
