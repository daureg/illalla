#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Put cities of the dataset on a world map, with size proportional to the
number of tweet using kartograph and data from Natural Earth site:
http://www.naturalearthdata.com/downloads/110m-physical-vectors/
"""
import sys
sys.path.append('..')
import cities as c
import fiona
import json
from shapely.geometry import mapping, box
from operator import itemgetter
if __name__ == '__main__':
    layers = {}
    size = 10
    schema = {'geometry': 'Polygon', 'properties': {'share': 'float',
                                                    'name': 'str'}}
    for sname, bbox in zip(c.SHORT_KEY, c.US+c.EU):
        layers[sname] = {"src": sname+'.shp', "labeling": {"key": 'name'}}
        print("#{}-label {{font-size: {}px}}".format(sname, size))
        with fiona.collection(sname+'.shp', "w",
                              "ESRI Shapefile", schema) as f:
            rec = [{'geometry': mapping(box(*itemgetter(1, 0, 3, 2)(bbox))),
                   'properties': {"share": size, "name": c.FULLNAMES[sname]}}]
            f.writerecords(rec)
    print(json.dumps(layers))
