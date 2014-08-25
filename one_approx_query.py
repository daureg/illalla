#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Output JSON result of one ground truth query using approx algo (almost like
worldwide.py actually)."""
import worldwide as ww
import json
import sys
from operator import itemgetter
import neighborhood as nb


if __name__ == '__main__':
    with open('static/ground_truth.json') as gt:
        ground_truth = json.load(gt)
    source, target, district = itemgetter(1, 2, 3)(sys.argv)
    features = ground_truth[district]["gold"][source]
    geo = nb.choose_query_region(features)
    res = ww.query_in_one_city(source, target, geo)
    # print(res)
    outjson = [
        {'pos': rank+1, 'metric': 'femd', 'dst': reg['dst'],
         'venues': list(reg['venues']),
         'geo': ww.venues_to_geojson(reg['venues'], target)}
        for rank, reg in enumerate(sorted(res, key=itemgetter('dst')))]
    filename = 'static/{}_{}_{}.json'.format(source, district, target)
    with open(filename, 'w') as f:
        json.dump(outjson, f, sort_keys=True, indent=2,
                  separators=(',', ': '))
