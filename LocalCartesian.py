#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""
A subset of LocalCartesian from geographiclib C++ version.

Convert (lat, lng) point to euclidian (x, y) coordinates where the origin is
the center of a city.
"""
from geographiclib.geodesic import Geodesic
EARTH = Geodesic.WGS84
from geographiclib.geomath import Math as geomath
from geographiclib.constants import Constants as geoconst
_a = geoconst.WGS84_a
_f = geoconst.WGS84_f
_e2 = _f * (2 - _f)
_e2m = 1 - _e2
import math
import numpy


class LocalCartesian(object):
    def __init__(self, lat0, lon0, h0=0):
        self.lat0 = lat0
        self.lon0 = geomath.AngNormalize(lon0)
        self.h0 = h0
        self.origin = (earth_forward(self.lat0, self.lon0, self.h0))
        phi = math.radians(lat0)
        lam = math.radians(lon0)
        sphi = math.sin(phi)
        cphi = 0 if abs(self.lat0) == 90 else math.cos(phi)
        slam = 0 if self.lon0 == -180 else math.sin(lam)
        clam = 0 if abs(self.lon0) == 90 else math.cos(lam)
        self.rot = geocentric_rotation(sphi, cphi, slam, clam)

    def forward(self, lat_lng, h=0):
        """Convert from geodetic to local cartesian coordinates"""
        if hasattr(lat_lng, 'shape'):
            column_dim = (lat_lng.shape[0], 1)
            lat_lng = numpy.hstack([lat_lng, numpy.tile(h, column_dim)])
            return ((bforward(lat_lng) - self.origin)*self.rot).A[:, :2]
        lat, lon = lat_lng
        result = (earth_forward(lat, lon, h) - self.origin)*self.rot
        return (result[0, :]).A1


def earth_forward(lat, lon, h):
    """Geocentric::IntForward"""
    lon = geomath.AngNormalize(lon)
    phi = math.radians(lat)
    lam = math.radians(lon)
    sphi = math.sin(phi)
    cphi = 0 if abs(lat) == 90 else math.cos(phi)
    _a = geoconst.WGS84_a
    _f = geoconst.WGS84_f
    _e2 = _f * (2 - _f)
    _e2m = 1 - _e2
    n = _a/math.sqrt(1 - _e2 * sphi * sphi)
    slam = 0 if lon == -180 else math.sin(lam)
    clam = 0 if abs(lon) == 90 else math.cos(lam)
    Z = (_e2m * n + h) * sphi
    X = (n + h) * cphi
    Y = X * slam
    X *= clam
    return numpy.array([X, Y, Z])


def bforward(coords):
    angles = to_unsafe_angles(coords)
    n = _a/numpy.sqrt(1 - _e2 * angles[:, 0] * angles[:, 0])
    Z = (_e2m * n + coords[:, 2]) * angles[:, 0]
    X = (n + coords[:, 2]) * angles[:, 1]
    Y = X * angles[:, 2]
    X *= angles[:, 3]
    return numpy.vstack([X, Y, Z]).T


def to_unsafe_angles(coords):
    """Return a n×4 array of (sphi, cphi, slam, clam) from a n×2 array
    `coords` of (lat, lon), without any validity checks or correction."""
    phi = numpy.radians(coords[:, 0])
    lam = numpy.radians(coords[:, 1])
    return numpy.vstack([numpy.sin(phi), numpy.cos(phi),
                         numpy.sin(lam), numpy.cos(lam)]).T


def to_unsafe_coords(angles):
    """Reverse the action of to_unsafe_angles."""
    lat = numpy.degrees(numpy.arcsin(angles[:, 0]))
    aslon = numpy.arcsin(angles[:, 2])
    lon = numpy.sign(aslon) * numpy.degrees(numpy.arccos(angles[:, 3]))
    return numpy.vstack([lat, lon]).T


def geocentric_rotation(sphi, cphi, slam, clam):
    """Geocentric::Rotation"""
    return numpy.matrix([
        [-slam,  clam, 0],
        [-clam * sphi, -slam * sphi, cphi],
        [clam * cphi,  slam * cphi, sphi]]).transpose()


if __name__ == '__main__':
    # ref value where obtain by
    # echo "60.15 24.91 0" | CartConvert -l 60.19415 24.92945 0
    ref = '-1080.3931874725 -4918.8217761851 -1.9863406491'.split()
    center = LocalCartesian(60.19415, 24.92945)
    res = center.forward([60.15, 24.91])
    print(res)
    print(ref)
    print(numpy.allclose(map(float, ref), res))
    tdata = numpy.array([(60.15, 24.91, 10),
                         (60.19415, 24.92945, 30),
                         (59, 25, 0)])
    print(center.forward(tdata).shape)
