#! /usr/bin/python2
# vim: set fileencoding=utf-8
POINTS = [
    [-122.381388, 37.775833],
    [-122.382084, 37.7771],
    [-122.382084, 37.7771],
    [-122.3825, 37.777777],
    [-122.382851, 37.776382],
    [-122.384166, 37.776666],
    [-122.381388, 37.775833],
    [-122.382084, 37.7771],
    [-122.382084, 37.7771],
    [-122.3825, 37.777777],
    [-122.382851, 37.776382],
    [-122.384166, 37.776666],
    [-122.381388, 37.775833],
    [-122.382084, 37.7771],
    [-122.382084, 37.7771],
    [-122.3825, 37.777777],
    [-122.382851, 37.776382],
    [-122.384166, 37.776666],
    [-122.381388, 37.775833],
    [-122.382084, 37.7771],
    [-122.382084, 37.7771],
    [-122.3825, 37.777777],
    [-122.382851, 37.776382],
    [-122.384166, 37.776666],
    [-122.381388, 37.775833],
    [-122.382084, 37.7771],
    [-122.382084, 37.7771],
    [-122.3825, 37.777777],
    [-122.382851, 37.776382],
    [-122.384166, 37.776666],
    [-122.381388, 37.775833],
    [-122.382084, 37.7771],
    [-122.382084, 37.7771],
    [-122.3825, 37.777777],
    [-122.382851, 37.776382],
    [-122.384166, 37.776666],
    [-122.381388, 37.775833],
    [-122.382084, 37.7771],
    [-122.382084, 37.7771],
    [-122.3825, 37.777777],
    [-122.382851, 37.776382],
    [-122.384166, 37.776666],
    [-122.385657, 37.777122]
]
from shapely import speedups
import shapely
from timeit import default_timer as clock


# TODO: generate 10000 random points in San Fransisco and save them in a disk
# matrix. Then benchmark shapely, scipy.spatial and matlab/octave in the three
# problems:
# - distribution from gravity
# - distribution of pairwise distance
# - distribution of nearest neighbour distance

def dst_to_gravity(points):
    centroid = points.centroid
    return map(lambda p: p.distance(centroid), list(points))


def pairwise_distance(points):
    dst = []
    for i, p in enumerate(points):
        dst.extend(map(lambda q: q.distance(p), points[i+1:]))
    return dst


if __name__ == '__main__':
    if speedups.available:
        speedups.enable()

    start = clock()
    pts = shapely.geometry.MultiPoint(map(lambda p: (p[1], p[0]), POINTS))
    t = 1000*(clock() - start)
    print(t)

    start = clock()
    dst = dst_to_gravity(pts)
    t = 1000*(clock() - start)
    print(t)

    start = clock()
    dst2 = pairwise_distance(pts)
    t = 1000*(clock() - start)
    print(t)
