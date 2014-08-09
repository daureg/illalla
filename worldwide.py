#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Find closest region in every other cities in the world."""
import cities as c
import json
import itertools
from scipy.spatial import cKDTree, ConvexHull
import approx_emd as app
import numpy as np
import neighborhood as nb
import persistent as p
import shapely.geometry as sgeo
from operator import itemgetter


# load venues location for all cities
print('start loading city info')
cities = set(c.SHORT_KEY)
cities_venues_raw = {name: p.load_var(name+'_svenues.my')
                     for name in cities}
cities_desc = {name: nb.cn.gather_info(name, raw_features=True,
                                       hide_category=True)
               for name in cities}
cities_venues = {}
cities_index = {}
cities_kdtree = {}
for city in cities:
    vids, _, locs = cities_venues_raw[city].all()
    vindex = cities_desc[city]['index']
    cities_venues[city] = np.zeros((len(vindex), 2))
    cities_index[city] = dict(itertools.imap(lambda x: (x[1], x[0]),
                                             enumerate(vindex)))
    for vid, loc in itertools.izip(vids, locs):
        pos = cities_index[city].get(vid)
        if pos is not None:
            cities_venues[city][pos, :] = loc
    cities_kdtree[city] = cKDTree(cities_venues[city])
print('done')


def retrieve_closest_venues(query_venues, query_city, target_city):
    """For the given query, return a list of venues indices for knn level of
    50"""
    mask = np.where(np.in1d(cities_desc[query_city]['index'], query_venues))[0]
    query_features = cities_desc[query_city]['features'][mask, :]
    all_target_features = cities_desc[target_city]['features']
    tindex = cities_desc[target_city]['index']
    candidates = app.get_candidates_venues(query_features,
                                           all_target_features, k=60)
    threshold = int(len(tindex)*1.0*len(query_venues) /
                    len(cities_desc[query_city]['index']))
    return candidates, threshold


def query_in_one_city(source, target, region):
    raw_result = []
    infos = nb.interpret_query(source, target, region, 'emd')
    _, right, _, regions_distance, vids, _ = infos
    vindex = np.array(right['index'])
    vloc = cities_venues[target]
    infos = retrieve_closest_venues(vids, source, target)
    candidates, _ = infos
    print(source, target)

    eps, mpts = 250, 10 if len(vloc) < 5000 else 40
    clusters = app.good_clustering(vloc, list(sorted(candidates)), eps, mpts)
    areas = []
    for cluster in clusters:
        venues_areas = app.cluster_to_venues(cluster, vloc,
                                             cities_kdtree[target], 4)
        if len(venues_areas) == 0:
            continue
        for venues in venues_areas:
            vids = vindex[venues]
            venues = right['features'][venues, :]
            dst = regions_distance(venues.tolist(),
                                   nb.weighting_venues(venues[:, 1]))
            areas.append({'venues': set(vids), 'dst': dst})
    res = [a['dst'] for a in areas]
    venues_so_far = set()
    for idx in np.argsort(res):
        cand = set(areas[idx]['venues'])
        if not venues_so_far.intersection(cand):
            venues_so_far.update(cand)
        else:
            continue
        raw_result.append(areas[idx])
        if len(raw_result) >= 5:
            break
    return raw_result


def venues_to_geojson(vids, city):
    """Convert a list of venues id into a GeoJSON polygon"""
    mask = itemgetter(*vids)(cities_index[city])
    locs = cities_venues[city][mask, :]
    hull = locs[ConvexHull(locs).vertices, :]
    geohull = c.euclidean_to_geo(city, hull)
    return sgeo.mapping(sgeo.Polygon(np.fliplr(geohull)))


if __name__ == '__main__':
    with open('thirdworld.json') as inf:
        regions = json.load(inf)
    from collections import namedtuple, defaultdict
    Query = namedtuple('Query', 'origin targets name geo'.split())
    queries = []
    for region in regions:
        origin = region['properties']['origin']
        targets = cities.difference([origin])
        name = region['properties']['name']
        geo = region['geometry']
        queries.append(Query(origin, targets, name, geo))

    results = defaultdict(list)
    for query in queries:
        this_query = {}
        for city in query.targets:
            for res in query_in_one_city(query.origin, city, query.geo):
                results[query.name].append((city, res['dst'], res['venues']))
