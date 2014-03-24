#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Compute discrepancy between tweets and photos."""
import scipy.io as sio
import numpy as np
import more_query as mq
from explore import Entity
import spatial_scan as sps
import matplotlib as mpl
FREQUENCE_FILE = 'freq_{}_{}_{}.mat'


def compute_frequency(client, city, entity, k=200):
    """Splits `city` in k^2 rectangles and save the frequency of `entity` in
    each of them as a matrix."""
    assert entity in [Entity.photo, Entity.checkin]
    freq_name = FREQUENCE_FILE.format(entity.name, city, k)
    try:
        mat_count = sio.loadmat(freq_name)['c']
        return mat_count
    except IOError:
        pass
    count = (k*k+1)*[0, ]
    get_data = {Entity.photo: get_photos, Entity.checkin: get_checkins}[entity]
    coords = get_data(client, city)
    bbox = (cities.US+cities.EU)[cities.INDEX[city]]
    rectangles, rect_to_index, _ = mq.k_split_bbox(bbox, k)
    # count[0] is for potential points that do not fall in any region (it
    # must only happens because of rounding inprecision)
    count = (len(rectangles)+1)*[0, ]
    for loc in coords:
        count[rect_to_index(loc)+1] += 1
    mat_count = np.array(count[1:])
    sio.savemat(freq_name, {'c': mat_count}, do_compression=True)
    return mat_count


def get_photos(client, city):
    """Return a list of [lng, lat] of all photos in `city`."""
    photos = client.world.photos.find({'hint': city}, {'loc': 1})
    return [p['loc']['coordinates'] for p in photos]


def get_checkins(client, city):
    """Return a list of [lng, lat] of all checkins in `city`."""
    checkins = client.foursquare.checkin.find({'city': city}, {'loc': 1})
    return [p['loc']['coordinates'] for p in checkins]


def output_json(regions, photos_as_background=True):
    """Write a GeoJSON collection of `regions` with their discrepancy."""
    discrepancies = [v[0] for v in regions]
    colormap = mpl.cm.ScalarMappable(sps.mcolor.Normalize(min(discrepancies),
                                                          max(discrepancies)),
                                     'YlOrBr')
    schema = {'geometry': 'Polygon', 'properties': [('discrepancy', 'float'),
                                                    ('color', 'str'),
                                                    ('photos', 'float'),
                                                    ('checkins', 'float')]}
    get_color = lambda v: sps.mcolor.rgb2hex(colormap.to_rgba(v))
    if photos_as_background:
        photos_idx, checkins_idx = 2, 3
    else:
        photos_idx, checkins_idx = 3, 2
    polys = [{'geometry': sps.mapping(r[1]), 'properties':
              {'discrepancy': r[0], 'color': get_color(r[0]),
               'photos': r[photos_idx], 'checkins': r[checkins_idx]}}
             for r in regions]
    name = 'paris_d.json'
    import os
    os.remove(name)
    print(polys[0])
    with sps.fiona.collection(name, "w", "GeoJSON", schema) as f:
        f.writerecords(polys)


def do_scan(client, city, k):
    """Perform discrepancy scan on `city` with grid_size."""
    photos = compute_frequency(client, city, Entity.photo, k)
    checkins = compute_frequency(client, city, Entity.checkin, k)
    background, measured = photos, checkins
    total_b = np.sum(background)
    total_m = np.sum(measured)
    if not total_m > 0:
        return
    if 0 < total_m <= 500:
        support = 20
    if 500 < total_m <= 2000:
        support = 40
    if 2000 < total_m:
        support = sps.MAX_SUPPORT
    discrepancy = sps.get_discrepancy_function(total_m, total_b, support)
    grid_dim = (k, k)
    info = u'g={}, s={}, k={}, w={}, h={}, max={}'
    print(info.format(k, support, sps.TOP_K, sps.MIN_WIDTH, sps.MIN_HEIGHT,
                      sps.MAX_SIZE))
    top_loc = sps.exact_grid(np.reshape(measured, grid_dim),
                             np.reshape(background, grid_dim),
                             discrepancy, sps.TOP_K, k/sps.MAX_SIZE)
    return top_loc

if __name__ == '__main__':
    #pylint: disable=C0103
    import CommonMongo as cm
    import arguments
    import cities
    args = arguments.city_parser().parse_args()
    city = args.city
    # _, client = cm.connect_to_db('foursquare', args.host, args.port)
    client = None
    k = 200
    sps.GRID_SIZE = k
    bbox = (cities.US+cities.EU)[cities.INDEX[city]]
    sps.BBOX = bbox
    _, _, sps.index_to_rect = sps.k_split_bbox(bbox, k)
    top_loc = do_scan(client, city, k)
    output_json(sps.merge_regions(top_loc))
