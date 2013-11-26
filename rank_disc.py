import persistent
from more_query import get_top_tags, CSS, KARTO_CONFIG, SF_BBOX
from shapely.geometry import mapping
from shapely import speedups
if speedups.available:
    speedups.enable()
import fiona
import json
from os.path import join as mkpath


def top_discrepancy(allowed_tags=None):
    return sorted([(v[0][0], v[0][1], k) for k, v in t.items()
                   if len(v) > 0 and (allowed_tags is None or
                                      k in allowed_tags)],
                  key=lambda x: x[0], reverse=True)


def plot_some(tags, n=15):
    """plot label of the n most discrepants tags"""
    schema = {'geometry': 'Polygon', 'properties': {'tag': 'str'}}
    style = []
    KARTO_CONFIG['bounds']['data'] = [SF_BBOX[1], SF_BBOX[0],
                                      SF_BBOX[3], SF_BBOX[2]]
    # TODO cluster polys by their area so label's size can depend of it
    polys = [{'geometry': mapping(r[1]),
              'properties': {'tag': r[2]}} for r in tags[:n]]
    name = u'top_disc'
    KARTO_CONFIG['layers'][name] = {'src': name+'.shp',
                                    'labeling': {'key': 'tag'}}
    color = 'red'
    style.append(CSS.format(name, color, 'black'))
    with fiona.collection(mkpath('disc', name+'.shp'),
                          "w", "ESRI Shapefile", schema) as f:
        f.writerecords(polys)

    with open(mkpath('disc', 'photos.json'), 'w') as f:
        json.dump(KARTO_CONFIG, f)
    with open(mkpath('disc', 'photos.css'), 'w') as f:
        f.write('\n'.join(style))

if __name__ == '__main__':
    t = persistent.load_var('disc/all')
    top = get_top_tags(500, 'nsf_tag.dat')
    supported = [v[0] for v in persistent.load_var('supported')][:600]
    d = top_discrepancy(supported)
    # print([v[2] for v in d[-15:]], [v[2] for v in d[:15]])
    plot_some(d)
