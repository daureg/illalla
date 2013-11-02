#! /usr/bin/python
# vim: set fileencoding=utf-8

# import izip_longest instead for python 2
from itertools import izip_longest as zip_longest
import codecs


def outplot(filename, colnames, *args):
    res = []
    res.append('\t'.join(colnames))
    for vals in zip_longest(*args):
        res.append('\t'.join(list(map(lambda x: u'{}'.format(x), vals))))
    with codecs.open(filename, 'w', 'utf8') as f:
        f.write('\n'.join(res))

if __name__ == '__main__':
    outplot('__test', ['x', 'y'], list(range(5)), list(map(lambda x: x*x,
                                                           range(5))))
    tags = [u'center', u'サンフランシスコ', u'buildings']
    val = [1, 2, 3]
    outplot('__unicode', ['tags', 'count'], tags, val)
