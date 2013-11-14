#! /usr/bin/python2
# vim: set fileencoding=utf-8
import more_query as mq
import datetime
import pymongo
import json
from timeit import default_timer as clock

if __name__ == '__main__':
    import sys
    HOST = 'lefalg1@kosh.aalto.fi:/u/99/lefalg1/data/flickr/illalla'
    tag = sys.argv[1]
    begin = clock()
    client = pymongo.MongoClient('localhost', 27017)
    DB = client['flickr']
    photos = DB['photos']
    SF_BBOX = [37.7123, -122.531, 37.84, -122.35]
    KARTO_CONFIG = mq.KARTO_CONFIG
    KARTO_CONFIG['bounds']['data'] = [SF_BBOX[1], SF_BBOX[0],
                                      SF_BBOX[3], SF_BBOX[2]]
#    KARTO_CONFIG['layers']['photos'] = {'src': '{}_1.shp'.format(tag)}
#    CSS = '#photos {fill: #0050ff; opacity: 0.8; stroke-width: 0;}'
#    with open('photos.json', 'w') as f:
#        json.dump(KARTO_CONFIG, f)
#    with open('photos.css', 'w') as f:
#        f.write(CSS)
    start = datetime.datetime(2008, 1, 1)
    end = datetime.datetime.now()
    # mq.tag_over_time(photos, tag, None, start, None)
    # mq.simple_metrics(photos, tag, SF_BBOX, start, end)
    e, kl = mq.compute_frequency(photos, tag, SF_BBOX, start, end, 200, plot=True)
    print('Entropy of {}: {:.7f}'.format(tag, e))
    print('cp {}_1.* {}_freq_* photos* karto/ && cd karto'.format(tag, tag))
    print('make photos.svg combine && mv sf.png {}.png'.format(tag))
    print('scp {}/karto/{}.png .'.format(HOST, tag))
    # print('scp {}/{}_pairwise.dat .'.format(HOST, tag))
    # print('look at bar.tex')
    t = 1000*(clock() - begin)
    print('done in {:.3f}ms'.format(t))
