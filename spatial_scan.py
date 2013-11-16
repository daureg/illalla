#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Implement exact algorithm of the paper: Spatial Scan Statistics:
Approximations and Performance Study."""
import heapq
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolor
from math import log
from more_query import SF_BBOX, KARTO_CONFIG, FIRST_TIME, LAST_TIME
CSS = '#{} {{fill: {}; opacity: 0.5; stroke: {}; stroke-width: 0.5px;}}'
from more_query import k_split_bbox, bbox_to_polygon, compute_frequency
from utils import to_css_hex
from shapely.geometry import shape, mapping
from shapely import speedups
if speedups.available:
    speedups.enable()
import fiona
import json
import scipy.io as sio
from ProgressBar import AnimatedProgressBar
import persistent
from os.path import join as mkpath
import pymongo
ALLD = []


def get_discrepancy_function(total_m, total_b, support):
    """Return a binary function computing the Kulldorff discrepancy, given
    that there is total_m measured data, total_b background data and we want
    at least support points (otherwise return None)."""
    def discrepancy(m, b):
        """Compute d(m, b) or return None if it lacks support."""
        if m < support or b < support:
            return None
        m_ratio = 1.0*m/total_m
        b_ratio = 1.0*b/total_b
        first_term = m_ratio*log(m_ratio/b_ratio)
        if abs(1-m_ratio) < 1e-7:
            # because 0 log 0 → 0 (np.allclose is probably better but 1000
            # times slower)
            second_term = 0
        else:
            second_term = (1-m_ratio)*log((1-m_ratio)/(1-b_ratio))
        if m_ratio*(1-b_ratio) > b_ratio*(1-m_ratio):
            return first_term + second_term
        else:
            return None
    return discrepancy


def add_maybe(new_value, values_so_far, max_nb_values):
    """Consider adding new_value to values_so_far if it is one of the largest
    max_nb_values."""
    real_value = new_value[0]
    if real_value is None:
        return values_so_far
    bottom_box = index_to_rect(new_value[1][0])
    upper_box = index_to_rect(new_value[1][1])
    total_box = bottom_box[:2] + upper_box[2:]
    poly = shape(bbox_to_polygon(total_box, False))
    new_value = (real_value, poly)
    ALLD.append(real_value)
    if len(values_so_far) == 0:
        return [new_value]
    if real_value > values_so_far[0][0]:
        if not all([(previous[1].disjoint(poly) or previous[1].touches(poly))
                    for previous in values_so_far]):
            return values_so_far
        if len(values_so_far) < max_nb_values:
            heapq.heappush(values_so_far, new_value)
        else:
            heapq.heappushpop(values_so_far, new_value)
    return values_so_far


def exact_grid(measured, background, discrepancy, nb_loc=1, max_size=5):
    """Given the two g×g arrays representing the measure of interest and the
    background data, find the nb_loc region that have the most discrepancy
    according to the provided binary function to compute it. Consider only
    region with side smaller than g/max_size."""
    assert np.size(measured) == np.size(background), "use same size input"
    grid_size = np.size(measured, 0)
    assert grid_size == GRID_SIZE, "GRID_SIZE conflict with provided data"
    side = grid_size/max_size
    max_values = []
    p = AnimatedProgressBar(end=grid_size, width=120)
    for i in range(grid_size):  # left line
        cum_m = np.cumsum(measured[i, :])
        cum_b = np.cumsum(background[i, :])
        p + 1
        p.show_progress()
        for j in range(i+1, min(grid_size, i+1+side)):  # right line
            cum_m += np.cumsum(measured[j, :])
            cum_b += np.cumsum(background[j, :])
            for k in range(grid_size):  # bottom line
                for l in range(k, min(grid_size, k+side)):  # top line
                    if k == 0:
                        m = cum_m[k]
                        b = cum_b[k]
                    else:
                        m = cum_m[l] - cum_m[k-1]
                        b = cum_b[l] - cum_b[k-1]
                    max_values = add_maybe([discrepancy(m, b),
                                            [i*grid_size + k,
                                             j*grid_size + l]],
                                           max_values, nb_loc)
    return sorted(max_values, key=lambda x: x[0], reverse=True)


def plot_regions(regions, bbox, tag):
    """Output one shapefile for each region (represented by its bottom left and
    upper right index in the grid) with color depending of its discrepancy."""
    discrepancies = [v[0] for v in regions]
    colormap = cm.ScalarMappable(mcolor.Normalize(min(discrepancies),
                                                  max(discrepancies)),
                                 'YlOrBr')
    schema = {'geometry': 'Polygon', 'properties': {}}
    style = []
    KARTO_CONFIG['bounds']['data'] = [SF_BBOX[1], SF_BBOX[0],
                                      SF_BBOX[3], SF_BBOX[2]]
    for i, r in enumerate(regions):
        color = to_css_hex(colormap.to_rgba(r[0]))
        name = u'disc_{}_{:03}'.format(tag, i+1)
        KARTO_CONFIG['layers'][name] = {'src': name+'.shp'}
        color = 'red'
        style.append(CSS.format(name, color, 'black'))
        with fiona.collection(mkpath('disc', name+'.shp'),
                              "w", "ESRI Shapefile", schema) as f:
            poly = {'geometry': mapping(r[1]), 'properties': {}}
            f.write(poly)

    with open(mkpath('disc', 'photos.json'), 'w') as f:
        json.dump(KARTO_CONFIG, f)
    with open(mkpath('disc', 'photos.css'), 'w') as f:
        f.write('\n'.join(style))


def spatial_scan(tag):
    """The main method loads the data from the disk (or compute them) and
    calls appropriate methods to find top discrepancy regions."""
    grid_size = GRID_SIZE
    background_name = 'freq_{}_{}.mat'.format(grid_size, '_background')
    measured_name = 'freq_{}_{}.mat'.format(grid_size, tag)
    res = []
    for tag, filename in [(None, background_name), (tag, measured_name)]:
        try:
            res.append(sio.loadmat(filename)['c'])
        except IOError:
            compute_frequency(get_connection(), tag, SF_BBOX, FIRST_TIME,
                              LAST_TIME, grid_size, plot=False)
            res.append(sio.loadmat(filename)['c'])
    background, measured = res

    total_b = np.sum(background)
    total_m = np.sum(measured)
    discrepancy = get_discrepancy_function(total_m, total_b, 100)
    grid_dim = (grid_size, grid_size)
    top_loc = exact_grid(np.reshape(measured, grid_dim),
                         np.reshape(background, grid_dim), discrepancy, 10, 50)
    print('\n')
    for v in top_loc:
        print('{:.4f}'.format(v[0]))
    plot_regions(top_loc, SF_BBOX, tag)


def get_connection():
    client = pymongo.MongoClient('localhost', 27017)
    return client['flickr']['photos']


GRID_SIZE = 200
rectangles, dummy, index_to_rect = k_split_bbox(SF_BBOX, GRID_SIZE)
if __name__ == '__main__':
    import sys
    tag = 'museum' if len(sys.argv) <= 1 else sys.argv[1]
    spatial_scan(tag)
    persistent.save_var('alld', ALLD)
