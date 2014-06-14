#! /usr/bin/env python2
# vim: set fileencoding=utf-8
"""Keep only best disjoint circles of all sized by reading raw output"""
import persistent as p
import numpy as np
import os
import cities
from collections import defaultdict


def close_circle(center1, radius1, center2, radius2):
    """Tell if two circles are close"""
    distance = np.sqrt((center1[0]-center2[0])**2+(center1[1]-center2[1])**2)
    return distance < 1.3*(radius1 + radius2)


def get_top_disjoint(candidates, topk=5):
    """Return `topk` disjoint circles with lowest distance in `candidates`."""
    ordered = sorted(candidates, key=lambda x: (x[0], -x[1]))
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
    distance, nb_venues, center, radius, metric = cell
    suffix = '_alt' if alt else ''
    center = cities.euclidean_to_geo(city, center)
    return {'geo': {'type': 'circle', 'center': center, 'radius': radius},
            'dst': distance, 'metric': metric+suffix, 'nb_venues': nb_venues,
            'pos': pos}

if __name__ == '__main__':
    # pylint: disable=C0103
    import ujson
    res = {city: defaultdict(list) for city in ['barcelona', 'sanfrancisco']}
    neighborhoods = ["triangle", "latin", "montmartre", "pigalle", "marais",
                     "official", "weekend", "16th"]
    for city in res.keys():
        for neighborhood in neighborhoods:
            for metric in ['jsd', 'emd']:
                subtop = []
                for output in [name for name in os.listdir('comparaison/')
                               if name.startswith(city+'_'+neighborhood) and
                               name.endswith(metric+'.my')]:
                    subtop.extend(p.load_var('comparaison/'+output))
                top = get_top_disjoint(subtop, 5)
                json = [to_json(city, x[1]+[metric], x[0]+1)
                        for x in enumerate(top)]
                res[city][neighborhood].extend(json)
    out_name = 'static/cmp_metrics.js'
    with open(out_name, 'w') as out:
        out.write('var TOPREG =' + ujson.dumps(res) + ';')
