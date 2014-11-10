from approx_emd import point_inside_poly
from scipy.spatial import ConvexHull, cKDTree
from shapely.geometry import Polygon, mapping
import bisect
import itertools
import math as m
import numpy as np
import random
import json
import cities as c
import ClosestNeighbor as cn
import persistent as p
import report_metrics_results as rmr


# http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python
class WeightedRandomGenerator(object):
    def __init__(self, weights):
        self.totals = []
        running_total = 0
        for w in weights:
            running_total += w
            self.totals.append(running_total)

    def next(self):
        rnd = random.random() * self.totals[-1]
        return bisect.bisect_right(self.totals, rnd)

    def __call__(self):
        return self.next()


def load_data(city):
    features = cn.load_matrix(city + '_fv.mat')
    density = features['v'][:, 4]
    weights = density + np.abs(density.min())
    venues_generator = WeightedRandomGenerator(weights)

    vids, _, locs = p.load_var(city+'_svenues.my').all()
    vindex = features['i']
    venues = np.zeros((len(vindex), 2))
    index = dict(itertools.imap(lambda x: (x[1], x[0]),
                                enumerate(vindex)))
    for vid, loc in itertools.izip(vids, locs):
        pos = index.get(vid)
        if pos is not None:
            venues[pos, :] = loc
    kdtree = cKDTree(venues)

    with open('static/ground_truth.json') as infile:
        gold_list = json.load(infile)

    return vindex, venues_generator, venues, kdtree, gold_list


def sample_venue(locs, density_generator=None):
    """Return a random venue location, potentially weighted by density"""
    if density_generator:
        return locs[density_generator.next()]
    return random.choice(locs)


def build_random_poly(center, sides=6):
    """Return a convex polygon around `center` coordinates with `sides`
    sides."""
    angle_step = 2*m.pi/sides
    vertices = np.zeros((sides, 2))
    max_radius = -1
    for i in range(sides):
        radius = random.randint(150, 600)
        if radius > max_radius:
            max_radius = radius
        angle = (i + random.uniform(-.16, .15))*angle_step
        vertices[i, :] = center + np.array([radius*m.cos(angle),
                                            radius*m.sin(angle)])
    poly = Polygon(ConvexHull(vertices).points)
    poly.center = center
    return poly, max_radius


def venue_list(poly, previous, vindex, kdtree, venues, size=600):
    """return a list of venues in `poly` or None if it's not a valid one"""
    if previous and previous.intersects(poly):
        return None
    ids_around = kdtree.query_ball_point(poly.center, size)
    if ids_around < 35:
        return
    border = np.array(poly.exterior.coords)
    real_ids = [vindex[idx] for idx in ids_around
                if point_inside_poly(border, venues[idx, :])]
    return real_ids if 20 < len(real_ids) < 750 else None


def mock_random_list(city, district, city_info):
    """Return a list of five random polygons with their DCG score"""
    vindex, venues_generator, venues, kdtree, gold_list = city_info
    gold = [set(_['properties']['venues'])
            for _ in gold_list[district]['gold'].get(city, [])]
    previous = None
    res, relevances = [], []
    while len(res) < rmr.RES_SIZE:
        center = sample_venue(venues, venues_generator)
        poly, size = build_random_poly(center, 6)
        vids = venue_list(poly, previous, vindex, kdtree, venues, size)
        if not vids:
            continue
        if previous is None:
            previous = poly
        else:
            previous = previous.union(poly)
        res.append((np.array(poly.exterior.coords), vids))
        relevances.append(rmr.relevance(set(vids), gold))
    score = np.sum((np.array(relevances)**2 - 1) / rmr.discount_factor)
    return res, score + rmr.LOWEST

if __name__ == '__main__':
    import sys
    import os
    NTEST = 20
    city, districts = sys.argv[1], []
    city_info = load_data(city)
    gold_list = city_info[-1]
    districts = sorted(gold_list.keys())[:2]
    try:
        os.mkdir('random')
    except OSError:
        pass
    for district in districts:
        distrib, best_score, best_region = [], 0, None
        for i in range(NTEST):
            regions, score = mock_random_list(city, district, city_info)
            if score > best_score:
                best_score, best_region = score, regions
            distrib.append(score)
        p.save_var('random/{}_{}.my'.format(city, district), distrib)
        outjson = [{
            'pos': rank+1, 'metric': 'random', 'dst': -1, 'venues': r[1],
            'geo': mapping(Polygon(np.fliplr(c.euclidean_to_geo(city, r[0]))))}
            for rank, r in enumerate(best_region)]
        filename = 'static/random_{}_{}.json'.format(city, district)
        with open(filename, 'w') as f:
            json.dump(outjson, f, sort_keys=True, indent=2,
                      separators=(',', ': '))
