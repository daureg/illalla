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
BBOX = SF_BBOX
CSS = '#{} {{fill: {}; opacity: 0.5; stroke: {}; stroke-width: 0.5px;}}'
from more_query import k_split_bbox, bbox_to_polygon, compute_frequency, clock
from utils import to_css_hex
from shapely.geometry import shape, mapping, Point
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
from multiprocessing import Pool
ALLD = []


def get_discrepancy_function(total_m, total_b, support):
    """Return a binary function computing the Kulldorff discrepancy, given
    that there is total_m measured data, total_b background data and we want
    at least support points (otherwise return None)."""
    def discrepancy(m, b):
        global Reject
        """Compute d(m, b) or return None if it lacks support."""
        assert m <= total_m, "common!"
        if m < support or b < support:
            Reject += 1
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
    new_value = (real_value, poly, new_value[2], new_value[3])
    ALLD.append(real_value)
    if len(values_so_far) == 0:
        return [new_value]
    if real_value > values_so_far[0][0]:
        # need_heapify = False
        # discard = False
        # for i, previous in enumerate(values_so_far):
        #     if poly.intersects(previous[1]) and not poly.touches(previous[1]):
        #         if poly.area > previous[1].area:
        #             del values_so_far[i]
        #             print('\ndelete\n')
        #             need_heapify = True
        #         else:
        #             discard = True
        #             break
        # if discard:
        #     return [new_value]
        # if not all([(previous[1].disjoint(poly) or previous[1].touches(poly))
        #             for previous in values_so_far]):
        #     return values_so_far
        # if need_heapify:
        #     heapq.heapify(values_so_far)
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
    min_width = MIN_WIDTH
    min_height = MIN_HEIGHT
    p = AnimatedProgressBar(end=grid_size, width=120)
    for i in range(grid_size):  # left line
        cum_m = np.cumsum(measured[i, :])
        cum_b = np.cumsum(background[i, :])
        p + 1
        # p.show_progress()
        for j in range(i+min_width-1, min(grid_size, i+1+side)):  # right line
            if min_width != 1 and j == i+min_width:
                cum_m += np.sum(np.cumsum(measured[i:i+min_width, :], 1), 0)
                cum_b += np.sum(np.cumsum(background[i:i+min_width, :], 1), 0)
            if j > i:
                cum_m += np.cumsum(measured[j, :])
                cum_b += np.cumsum(background[j, :])
            for k in range(grid_size):  # bottom line
                for l in range(k+min_height-1, min(grid_size, k+side)):  # top line
                    if k == 0:
                        m = cum_m[l]
                        b = cum_b[l]
                    else:
                        m = cum_m[l] - cum_m[k-1]
                        b = cum_b[l] - cum_b[k-1]
                    max_values = add_maybe([discrepancy(m, b),
                                            [i*grid_size + k,
                                             j*grid_size + l], b, m],
                                           max_values, nb_loc)
    return sorted(max_values, key=lambda x: x[0], reverse=True)


def plot_regions(regions, bbox, tag):
    """Output one shapefile for each region (represented by its bottom left and
    upper right index in the grid) with color depending of its discrepancy."""
    #TODO not unicode safe
    discrepancies = [v[0] for v in regions]
    colormap = cm.ScalarMappable(mcolor.Normalize(min(discrepancies),
                                                  max(discrepancies)),
                                 'YlOrBr')
    schema = {'geometry': 'Polygon', 'properties': {}}
    style = []
    KARTO_CONFIG['bounds']['data'] = [BBOX[1], BBOX[0],
                                      BBOX[3], BBOX[2]]

    polys = [{'geometry': mapping(r[1]), 'properties': {}} for r in regions]
    for i, r in enumerate(regions):
        color = to_css_hex(colormap.to_rgba(r[0]))
        name = u'disc_{}_{:03}'.format(tag, i+1)
        KARTO_CONFIG['layers'][name] = {'src': name+'.shp'}
        color = 'red'
        style.append(CSS.format(name, color, 'black'))
        # style.append(CSS.format(name, color, color))
        with fiona.collection(mkpath('disc', name+'.shp'),
                              "w", "ESRI Shapefile", schema) as f:
            poly = {'geometry': mapping(r[1]), 'properties': {}}
            # f.write(poly)
            f.writerecords(polys)
        break

    with open(mkpath('disc', 'photos.json'), 'w') as f:
        json.dump(KARTO_CONFIG, f)
    with open(mkpath('disc', 'photos.css'), 'w') as f:
        f.write('\n'.join(style))


def spatial_scan(tag):
    """The main method loads the data from the disk (or compute them) and
    calls appropriate methods to find top discrepancy regions."""
    print(tag)
    grid_size = GRID_SIZE
    background_name = u'mfreq/freq_{}_{}.mat'.format(grid_size, '_background')
    measured_name = u'mfreq/freq_{}_{}.mat'.format(grid_size, tag)
    res = []
    for tag, filename in [(None, background_name), (tag, measured_name)]:
        try:
            res.append(sio.loadmat(filename)['c'])
        except IOError:
            conn, client = get_connection()
            compute_frequency(conn, tag, BBOX, FIRST_TIME,
                              LAST_TIME, grid_size, plot=False)
            client.close()
            res.append(sio.loadmat(filename)['c'])
    background, measured = res

    total_b = np.sum(background)
    total_m = np.sum(measured)
    if not total_m > 0:  # photos taken before 2008
        return
    if 0 < total_m <= 500:
        support = 20
    if 500 < total_m <= 2000:
        support = 40
    if 2000 < total_m:
        support = MAX_SUPPORT
    discrepancy = get_discrepancy_function(total_m, total_b, support)
    grid_dim = (grid_size, grid_size)
    top_loc = exact_grid(np.reshape(measured, grid_dim),
                         np.reshape(background, grid_dim),
                         discrepancy, TOP_K, GRID_SIZE/MAX_SIZE)
    info = u'\n{}: g={}, s={}, k={}, w={}, h={}, max={}'
    print(info.format(tag, GRID_SIZE, support, TOP_K, MIN_WIDTH, MIN_HEIGHT,
                      MAX_SIZE))
    # persistent.save_var(u'disc/top_{}'.format(tag), top_loc)
    # print('\n')
    # for v in top_loc:
    #     print('{:.4f}'.format(v[0]))
    #     print(top_loc[0][1].intersects(v[1]))
    plot_regions(merge_regions(top_loc), BBOX, tag)


def merge_regions(top_loc):
    merged = []
    merging = 0
    # print('from {}'.format(len(top_loc)))
    for i, loc in enumerate(top_loc):
        val, poly, back, meas = loc
        new_val = [val]
        new_back = [back]
        new_meas = [meas]
        # size = poly.area
        merged_neighbors = 0
        to_remove = []
        for j, others in enumerate(top_loc[i+1:]):
            if poly.intersects(others[1]):
                # inter = poly.intersection(others[1])
                # print(size, inter.area)
                # TODO: mean value, max value, linear combination with surface
                # coefficient
                # if poly.touches(others[1]) and merged_neighbors < 1:
                if merged_neighbors < 2:
                    poly = poly.union(others[1])
                    new_val.append(others[0])
                    new_back.append(others[2])
                    new_meas.append(others[3])
                    merged_neighbors += 1
                #     val = 0.5*(val + others[0])
                #     size = poly.area
                to_remove.append(j+i+1)
        merging += merged_neighbors
        for j in to_remove[::-1]:
            del top_loc[j]
        merged.append((np.mean(new_val), poly, np.mean(new_back),
                       np.mean(new_meas)))
    # print('to {} with {} merges'.format(len(merged), merging))
    return merged


def get_connection():
    client = pymongo.MongoClient('localhost', 27017)
    return client['flickr']['photos'], client


def post_process(tag):
    top_loc = persistent.load_var(u'disc/top_{}_{}'.format(tag, GRID_SIZE))
    merged = merge_regions(top_loc)
    persistent.save_var(u'disc/post_{}_{}'.format(tag, GRID_SIZE), merged)


def consolidate(tags):
    import os
    d = {tag: persistent.load_var(u'disc/post_{}_{}'.format(tag, GRID_SIZE))
         for tag in tags}
    persistent.save_var(u'disc/all_{}'.format(GRID_SIZE), d)


def get_best_tags(point):
    tags = persistent.load_var(u'disc/all_{}'.format(GRID_SIZE), d)
    res = []
    size = point.area
    for tag, polys in tags.items():
        for val, poly in polys:
            if point.intersects(poly) and \
               point.intersection(poly).area > .6*size:
                res.append((tag, val))
                break
    return sorted(res, key=lambda x: x[1], reverse=True)

GRID_SIZE = 80
TOP_K = 2000
MIN_WIDTH = 1
MIN_HEIGHT = 1
MAX_SIZE = 4
MAX_SUPPORT = 250
Reject = 0
rectangles, dummy, index_to_rect = k_split_bbox(BBOX, GRID_SIZE)
if __name__ == '__main__':
    import sys
    import random
    # random.seed(135)
    tag = 'museum' if len(sys.argv) <= 1 else sys.argv[1]
    tt = clock()
    # tmp = persistent.load_var('supported')
    # tags = [v[0] for v in tmp]
    # random.shuffle(tags)
    # tt = clock()
    # p = Pool(3)
    # p.map(spatial_scan, tags)
    # p.map(post_process, tags)
    # p.close()
    # consolidate(tags)
    # print(get_best_tags(Point(-122.409615, 37.7899132)))
    spatial_scan(tag)
    # sio.savemat('alld', {'d': ALLD})
    print(Reject)
    print('done in {:.2f}.'.format(clock() - tt))
    # plot_regions(merged, BBOX, tag)
    # persistent.save_var('alld', ALLD)
