#! /usr/bin/env python
# vim: set fileencoding=utf-8
from worldwide import query_in_one_city, venues_to_geojson
import json
import argparse

if __name__ == '__main__':
    # read arg to find input file
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input JSON file')
    args = parser.parse_args()
    with open(args.input) as f:
        query = json.load(f)
    # extract source, target, geo, venues list, 
    source = query['properties']['source']
    target = query['properties']['target']
    name = query['properties']['name']
    geo = query['geometry']
    venues = None
    if 'venues' in query['properties']:
        tmp = query['properties']['venues']
        if isinstance(tmp, list) and len(tmp) > 0:
            venues = tmp
    # run it and save output in normalized file
    best = query_in_one_city(source, target, geo, venues)[0]
    import persistent as p
    p.save_var('best.my', best)
    # best = p.load_var('best.my')
    # print(best)
    res = {"type": "Feature",
           "properties": {"source": source, "target": target,
                          "name": name, "venues": sorted(best['venues']),
                          "emd_distance": best['dst']},
           "geometry": venues_to_geojson(best['venues'], target)}
    output = '{}_in_{}_to_{}.json'.format(name, source, target)
    with open(output, 'w') as f:
        json.dump(res, f)
    print(output)
