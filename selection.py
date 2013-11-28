#! /usr/bin/python2
# vim: set fileencoding=utf-8
from bottle import route, run, static_file, post, request
from more_query import get_top_tags, SF_BBOX as bbox
import persistent
from spatial_scan import get_best_tags, GRID_SIZE as k
from shapely.geometry import Point, box
from rank_disc import js_some, top_discrepancy


@route('/<filename:re:.*\.js>')
def send_js(filename):
    return static_file(filename, root='.', mimetype='text/javascript')


@route('/disc/<filename:re:.*\.png>')
def send_image(filename):
    return static_file(filename, root='./disc', mimetype='image/png')


@route('/')
def index():
    return static_file('display.html', root='.')


@route('/cover')
def cover_map():
    return static_file('cover.html', root='.')


@post('/click')
def do_login():
    # x =  float(request.POST.get('x'))
    # y =  float(request.POST.get('y'))
    bx, by, tx, ty = [float(v) for v in request.POST.getlist('coords[]')]
    long_step = (bbox[3] - bbox[1])/k
    lat_step = (bbox[2] - bbox[0])/k
    x0 = bbox[1]
    y0 = bbox[0]
    bx, tx = (bx, tx) if tx > bx else (tx, bx)
    by, ty = (by, ty) if ty > by else (ty, by)
    return '\n'.join([u'{}: {:4f}'.format(t, float(v))
                      for t, v in get_best_tags(box(x0+bx*long_step,
                                                    y0+by*lat_step,
                                                    x0+tx*long_step,
                                                    y0+ty*lat_step))])
                      # for t, v in get_best_tags(Point(x0+x*long_step,
                      #                                 y0+y*lat_step))])


@post('/cover')
def cover():
    n = int(request.POST.get('n'))
    w = int(request.POST.get('w'))
    h = int(request.POST.get('h'))
    return js_some(d, n, w, h)

t = persistent.load_var('disc/all')
top = get_top_tags(500, 'nsf_tag.dat')
supported = [v[0] for v in persistent.load_var('supported')][:600]
d = top_discrepancy(t, supported)
run(host='localhost', port=8080, debug=True, reloader=True)
