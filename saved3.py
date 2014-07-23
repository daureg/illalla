#! /usr/bin/python2
# vim: set fileencoding=utf-8
import bottle as b
b.BaseRequest.MEMFILE_MAX = 5 * 1024 * 1024


@b.route('/')
def index():
    return b.static_file('scatter.html', root='.')


@b.route('/<filename:re:.*\.*>')
def send_js(filename):
    return b.static_file(filename, root='.', mimetype='text/javascript')


@b.post('/out')
def out():
    with open('d3out.xml', 'w') as f:
        f.write(b.request.POST.get('xml'))
    return "ok"

if __name__ == '__main__':
    b.run(host='localhost', port=8087, debug=True, reloader=True)
