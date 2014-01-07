import persistent
from more_query import get_top_tags, CSS, KARTO_CONFIG, SF_BBOX
from shapely.geometry import mapping, Polygon, box
from shapely.ops import polygonize
from shapely import speedups
if speedups.available:
    speedups.enable()
import fiona
import json
from os.path import join as mkpath


def coords_to_canvas_pixel(coords, canvas_w=2000, canvas_h=1758):
    res = []
    for i in range(0, len(coords), 2):
        lon = coords[i]
        lat = coords[i+1]
        res.append([(lon - SF_BBOX[1])*canvas_w/(1.0*SF_BBOX[3] - SF_BBOX[1]),
                    canvas_h*(1-(lat - SF_BBOX[0])/(1.0*SF_BBOX[2] -
                                                    SF_BBOX[0]))])
    return [int(v) for pair in res for v in pair]


def top_discrepancy(t, allowed_tags=None):
    return sorted([(v[0][0], v[0][1], k) for k, v in t.items()
                   if len(v) > 0 and (allowed_tags is None or
                                      k in allowed_tags)],
                  key=lambda x: x[0], reverse=True)


def js_some(tags, n=15, cw=1350, ch=1206, padding=0.1, overlap=True):
    # res = ['function topone(ctx, padding) {']
    res = []
    call = u'fit_text(ctx, {}, {}, {}, {}, "{}", {});'
    cover = None
    for r in tags[:n]:
        if (overlap) or (cover is None) or (not r[1].intersects(cover) or
                                            r[1].touches(cover)):
            info = coords_to_canvas_pixel(r[1].bounds, cw, ch)+[r[2], padding]
            res.append(call.format(*info))
            cover = r[1] if cover is None else cover.union(r[1])
    # res.append('}')
    with open('topone.js', 'w') as f:
        f.write('\n'.join(res))
    return '\n'.join(res)


def plot_some(tags, n=15):
    """plot label of the n most discrepants tags"""
    schema = {'geometry': 'Polygon', 'properties': {'tag': 'str'}}
    style = []
    KARTO_CONFIG['bounds']['data'] = [SF_BBOX[1], SF_BBOX[0],
                                      SF_BBOX[3], SF_BBOX[2]]
    # TODO cluster polys by their area so label's size can depend of it
    # polys = [{'geometry': mapping(r[1]),
    #           'properties': {'tag': r[2]}} for r in tags[:n]]
    polys = []
    cover = None
    for r in tags[:n]:
        diff = r[1] if cover is None else r[1].difference(cover)
        polys.append({'geometry': mapping(diff),
                      'properties': {'tag': r[2]}})
        cover = r[1] if cover is None else cover.union(r[1])
    name = u'top_disc'
    KARTO_CONFIG['layers'][name] = {'src': name+'.shp',
                                    'labeling': {'key': 'tag'}}
    color = '#ffa873'
    style.append(CSS.format(name, color, 'black'))
    with fiona.collection(mkpath('disc', name+'.shp'),
                          "w", "ESRI Shapefile", schema) as f:
        f.writerecords(polys)

    with open(mkpath('disc', 'photos.json'), 'w') as f:
        json.dump(KARTO_CONFIG, f)
    style.append('#top_disc-label {font-family: OpenSans; font-size: 14px}')
    with open(mkpath('disc', 'photos.css'), 'w') as f:
        f.write('\n'.join(style))
    sf = box(SF_BBOX[1], SF_BBOX[0], SF_BBOX[3], SF_BBOX[2])
    print(sf.bounds)
    print(100*cover.area, sf.area)

if __name__ == '__main__':
    t = persistent.load_var('disc/all')
    # top = get_top_tags(2000, 'nsf_tag.dat')
    supported = [v[0] for v in persistent.load_var('supported')][:600]
    d = top_discrepancy(t, supported)
    N=11
    print([v[2] for v in d[-N:]], [v[2] for v in d[:N]])
    # plot_some(d, 20)
    # js_some(d, 15)
