#! /usr/bin/env python2
# vim: set fileencoding=utf-8
"""Keep only best disjoint circles of all sized by reading raw output"""
import persistent as p
import numpy as np
import os
import sys
import cities
from collections import defaultdict


def close_circle(center1, radius1, center2, radius2):
    """Tell if two circles are close"""
    distance = np.sqrt((center1[0]-center2[0])**2+(center1[1]-center2[1])**2)
    return distance < 1.3*(radius1 + radius2)


def get_top_disjoint(candidates, topk=5):
    """Return `topk` disjoint circles with lowest distance in `candidates`."""
    list_size = lambda x: len(x) if isinstance(x, list) else x
    ordered = sorted(candidates, key=lambda x: (x[0], -list_size(x[1])))
    if not ordered:
        return []
    top = [ordered[0]]
    for cell in ordered[1:]:
        is_close = lambda reg: close_circle(reg[-2], reg[-1],
                                            cell[-2], cell[-1])
        if not any(map(is_close, top)):
            top.append(cell)
        if len(top) == topk:
            break
    return top


def to_json(city, cell, pos, alt=False):
    """Convert a `cell` ranked `pos` in `city` to a GeoJSONish dict"""
    distance, venues, center, radius, metric = cell
    suffix = '_alt' if alt else ''
    center = cities.euclidean_to_geo(city, center)
    return {'geo': {'type': 'circle', 'center': center, 'radius': radius},
            'dst': distance, 'metric': metric+suffix, 'venues': venues,
            'pos': pos}


CITIES = ['barcelona', 'sanfrancisco', 'rome', 'newyork', 'washington', 'paris']
# CITIES = ['barcelona']
NEIGHBORHOODS = ["triangle", "latin", "montmartre", "pigalle", "marais",
                 "official", "weekend", "16th"]
# NEIGHBORHOODS = ['triangle', 'latin']
# METRICS = ['jsd', 'emd', 'cluster', 'emd-lmnn', 'leftover']
METRICS = ['emd-itml', 'emd-tsne']
if __name__ == '__main__':
    # pylint: disable=C0103
    import json
    query_city = sys.argv[1]
    assert query_city in CITIES, ', '.join(CITIES)
    CITIES.remove(query_city)
    input_dir = 'www_comparaison_' + query_city
    res = {city: defaultdict(list) for city in CITIES}
    for city in res.keys():
        for neighborhood in NEIGHBORHOODS:
            for metric in METRICS:
                subtop = []
                for output in [name for name in os.listdir(input_dir)
                               if name.startswith(city+'_'+neighborhood) and
                               name.endswith(metric+'.my')]:
                    output = os.path.join(input_dir, output)
                    subtop.extend(p.load_var(output))
                top = get_top_disjoint(subtop, 5)
                if not top:
                    continue
                json_cell = [to_json(city, x[1]+[metric], x[0]+1)
                             for x in enumerate(top)]
                res[city][neighborhood].extend(json_cell)
    out_name = 'static/www_cmp_{}.js'.format(query_city)
    with open(out_name, 'w') as out:
        out.write('var TOPREG =\n' + json.dumps(res, sort_keys=True, indent=2,
                                                separators=(',', ': ')) + ';')
