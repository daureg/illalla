#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Compute discrepancy between tweets and photos."""
import scipy.io as sio
import numpy as np
import more_query as mq
from explore import Entity
import spatial_scan as sps
import matplotlib as mpl
import os
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


def full_disc_json(lratio, nz):
    it = np.nditer(lratio, flags=['f_index'])
    colormap = mpl.cm.ScalarMappable(sps.mcolor.Normalize(nz.min(), nz.max()),
                                     'coolwarm')
    schema = {'geometry': 'Polygon', 'properties': [('ratio', 'float'),
                                                    ('color', 'str')]}
    get_color = lambda v: sps.mcolor.rgb2hex(colormap.to_rgba(v))
    polys = []
    box = lambda i: sps.shape(sps.bbox_to_polygon(sps.index_to_rect(i), False))
    while not it.finished:
        idx, val = it.index, it[0]
        if not np.isinf(val):
            val = float(val)
            polys.append({'geometry': sps.mapping(box(idx)), 'properties':
                          {'ratio': val, 'color': get_color(val)}})
        it.iternext()
    print(polys[0])
    name = 'maps/paris_full_d.json'
    write_collection(polys, name, schema)


def write_collection(polys, name, schema):
    """Write JSON array `polys` in the file `name` using `schema`."""
    try:
        os.remove(name)
    except OSError:
        pass
    print(name)
    with sps.fiona.collection(name, "w", "GeoJSON", schema) as f:
        f.writerecords(polys)


def output_json(regions, options):
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
    city = options['city']
    photos_as_background = options['photos_background']
    prefix = '_only' if options['only'] else ''
    ratio = options.pop('ratio', 0)
    if photos_as_background:
        photos_idx, checkins_idx = 2, 3
        name = city+prefix+'_checkins_d.json'
        options['photos_ratio'] = ratio
    else:
        photos_idx, checkins_idx = 3, 2
        name = city+prefix+'_photos_d.json'
        options['checkins_ratio'] = ratio
    polys = [{'geometry': sps.mapping(r[1]), 'properties':
              {'discrepancy': r[0], 'color': get_color(r[0]),
               'photos': r[photos_idx], 'checkins': r[checkins_idx]}}
             for r in regions]
    name = os.path.join('maps', name)
    write_collection(polys, name, schema)
    options['city'] = '"{}"'.format(options['city'])
    with open(os.path.join('maps', city+'.js'), 'a') as f:
        f.write('\n'.join(['var {} = {};'.format(var, str(val).lower())
                           for var, val in options.iteritems()]))
    options['city'] = city


def compute_ratio(background, measured):
    """Compute the mean ratio of non extrem values between `background` and
    `measured` where both occured."""
    both = np.logical_and(background > 0, measured > 0)
    background = background.astype(np.float32, copy=False)
    measured = measured.astype(np.float32, copy=False)
    ratio = background[both]/measured[both]
    lower, upper = np.percentile(ratio, [5, 95])
    return np.mean(ratio[np.logical_and(ratio >= lower, ratio <= upper)])


def load_frequency(client, city, k, photos_as_background=True):
    """Set background and measured array in the right order"""
    photos = compute_frequency(client, city, Entity.photo, k)
    checkins = compute_frequency(client, city, Entity.checkin, k)
    if photos_as_background:
        return photos, checkins
    return checkins, photos


def do_scan(client, city, k, photos_as_background=True):
    """Perform discrepancy scan on `city` with grid_size."""
    background, measured = load_frequency(client, city, k,
                                          photos_as_background)
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
    return top_loc, compute_ratio(background, measured)


def stand_alone(client, city, k, photos_as_background=True, p=99.5):
    """Return the top list of cells with only one kind of entity."""
    background, measured = load_frequency(client, city, k,
                                          photos_as_background)
    alone = np.logical_and(measured > 0, background < 1)
    threshold = np.percentile(measured[alone], p)
    high_alone = np.logical_and(alone, measured > threshold)
    print(photos_as_background)
    print(np.sum(measured))
    print(threshold)
    print(np.sum(high_alone))
    cells = np.argwhere(high_alone)[:, 1]
    count = measured[0, cells]
    max_values = []
    for idx, val in zip(cells, count):
        max_values = sps.add_maybe([val, [idx, idx], 0, val], max_values, 500)
    return sorted(max_values, key=lambda x: x[0], reverse=True)


if __name__ == '__main__':
    #pylint: disable=C0103
    import CommonMongo as cm
    import arguments
    import cities
    args = arguments.city_parser().parse_args()
    city = args.city
    _, client = cm.connect_to_db('foursquare', args.host, args.port)
    # client = None
    photos_in_background = True
    k = 200
    sps.GRID_SIZE = k
    sps.MAX_SUPPORT = 250
    bbox = (cities.US+cities.EU)[cities.INDEX[city]]
    sps.BBOX = bbox
    _, _, sps.index_to_rect = sps.k_split_bbox(bbox, k)
    options = {'city': city, 'photos_background': True,
               'bbox': cities.bbox_to_polygon(bbox), 'only': False}
    # top_loc, ratio = do_scan(client, city, k, options['photos_background'])
    # options['ratio'] = ratio
    # output_json(sps.merge_regions(top_loc), options)
    # options['photos_background'] = False
    # top_loc, ratio = do_scan(client, city, k, options['photos_background'])
    # options['ratio'] = ratio
    # output_json(sps.merge_regions(top_loc), options)

    # options['only'] = True
    # for pb in [True, False]:
    #     options['photos_background'] = pb
        # top_loc = stand_alone(client, city, 100,
        #                       options['photos_background'])
    #     output_json(sps.merge_regions(top_loc, use_mean=False), options)
