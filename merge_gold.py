#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Merge regions in the same city and info from DB."""


import ClosestNeighbor as cn
import pymongo
import json
import shapely.geometry as sgeo
import shapely.ops as sops
import cities as c
import numpy as np


def venues_within_geo(geo, vindex, db):
    """Return a list of venues, belonging to a predefined `vindex` list and
    contained within `geo`, by quering `db`."""
    real_geo = geo.get('geometry', geo.get('geo', None))
    if not real_geo:
        raise ValueError('cant read geometry from ' + str(geo))
    if real_geo['type'] == 'circle':
        center = {'type': 'Point', 'coordinates': real_geo['center']}
        ball = {'$geometry': center, '$maxDistance': real_geo['radius']}
        req = {'loc': {'$near': ball}}
    else:
        req = {'loc': {'$geoWithin': {'$geometry': real_geo}}}
    req['_id'] = {"$in": vindex}
    return [_['_id'] for _ in db.venue.find(req, {'_id': 1})]


def circle_to_poly(city, geo, resolution=4):
    """Discretize a circle (LngLat) to a Shapely polygon (LatLng)"""
    center, radius = geo['center'], geo['radius']
    local_center = c.GEO_TO_2D[city](list(reversed(center)))
    approx = sgeo.Point(*local_center).buffer(radius, resolution)
    local_exterior = c.euclidean_to_geo(city, np.array(approx.exterior.coords))
    return sgeo.Polygon(np.fliplr(local_exterior).tolist())


def merge_regions(city, district, db, city_venues):
    """Return a consolidated list of ground truth for `district` in `city`
    with contained venues."""
    raw = []
    for zone in regions[district]['gold'][city]:
        geo = zone['geometry']
        if geo['type'] == 'Polygon':
            raw.append(sgeo.shape(geo))
        else:
            raw.append(circle_to_poly(city, geo))
    merged = sops.unary_union(raw)
    if isinstance(merged, sgeo.Polygon):
        geoms_list = [merged]
    else:
        geoms_list = merged.geoms
    res = []
    for zone in geoms_list:
        geo = sgeo.mapping(zone)
        venues = venues_within_geo({"geo": geo}, city_venues, db)
        area = {"type": "Feature", "properties": {"venues": venues},
                "geometry": geo}
        res.append(area)
    return res
    # If the `res` array is wrapped in a FeatureCollection, it might be easier
    # to edit with http://geojson.io/ (but then I need to split into one file
    # per (city, district)
    # coll = {"type": "FeatureCollection", "features": res}
    # return coll

if __name__ == '__main__':
    # pylint: disable=C0103
    client = pymongo.MongoClient()
    db = client.foursquare
    with open('static/raw_ground_truth.json') as inf:
        regions = json.load(inf)
    cities_venues = {}
    for district, gold in regions.iteritems():
        # a not very elegant indirection
        gold = gold['gold']
        for city, areas in gold.iteritems():
            print(city, district)
            if city not in cities_venues:
                try:
                    city_venues = list(cn.load_matrix(city)['i'])
                except IOError:
                    city_venues = None
                cities_venues[city] = city_venues
            if cities_venues[city]:
                ground_truth = merge_regions(city, district, db,
                                             cities_venues[city])
                msg = '{}, {}: merged into {} areas'
                print(msg.format(city, district, len(ground_truth)))
                regions[district]['gold'][city] = ground_truth
            else:
                msg = '{}, {}: not merged'
                print(msg.format(city, district))
    with open('static/ground_truth.json', 'w') as out:
        json.dump(regions, out, sort_keys=True, indent=2,
                  separators=(',', ': '))
