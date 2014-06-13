#! /usr/bin/python2
# vim: set fileencoding=utf-8
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
    distance, nb_venues, center, radius = cell
    suffix = '_alt' if alt else ''
    center = cities.euclidean_to_geo(city, center)
    return {'geo': {'type': 'circle', 'center': center, 'radius': radius},
            'dst': distance, 'metric': metric+suffix, 'nb_venues': nb_venues,
            'pos': pos}

if __name__ == '__main__':
    # pylint: disable=C0103
    import arguments
    import ujson
    args = arguments.city_parser().parse_args()
    res = defaultdict(list)
    city = args.city
    for neighborhood in ["triangle", "latin", "montmartre", "pigalle",
                         "marais", "official", "weekend", "16th"]:
        for metric in ['jsd', 'emd']:
            subtop = []
            for output in [name for name in os.listdir('comparaison/')
                           if name.startswith(city+'_'+neighborhood) and
                           name.endswith(metric+'.my')]:
                subtop.extend(p.load_var('comparaison/'+output))
            top = get_top_disjoint(subtop, 5)
            res[neighborhood].extend(map(lambda x: to_json(city, x[1], x[0]+1),
                                         enumerate(top)))
    with open('static/cmp_metrics.js', 'w') as out:
        out.write('var TOPREG =' + ujson.dumps(res) + ';')
