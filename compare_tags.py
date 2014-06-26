#! /usr/bin/python2
# vim: set fileencoding=utf-8
import codecs
try:
    from collections import OrderedDict
except ImportError:
    from OrderedDict import OrderedDict


def read_top_tags(n=200, filename='sftags.dat'):
    with codecs.open(filename, 'r', 'utf8') as f:
        lines = [i.strip().split() for i in f.readlines()[1:n+1]]
        # see http://stackoverflow.com/a/1747827 for python 2.7+
        tags = OrderedDict((tag, int(count)) for tag, count in lines)
    return tags


def added_and_removed(before, after, n=500):
    before = read_top_tags(10*n, before)
    after = read_top_tags(n, after)
    before_tags = before.keys()
    tab = []
    import prettytable as pt
    t = pt.PrettyTable(['status', 'tag', 'move', 'rchange', 'fchange'])
    t.align['status'] = 'l'
    t.padding_width = 0
    for pos, tag in enumerate(after.keys()):
        if tag in before_tags:
            old_pos = before_tags.index(tag)
            if pos == old_pos:
                status = '='
                symb = ''
            else:
                status = '+' if pos < old_pos else '-'
                symb = u'↑' if pos < old_pos else u'↓'
            pdiff = pos - old_pos
            cdiff = after[tag] - before[tag]
            rdiff = 100.0*cdiff/before[tag]
            t.add_row([status, tag, u'{}{}'.format(abs(pdiff), symb),
                       '{:.2f}%'.format(rdiff), cdiff])
        else:
            tab.append(('+', tag, 'N/A', 'N/A'))
    return t.get_string(border=False, left_padding_width=0,
                        right_padding_width=2)

if __name__ == '__main__':
    with codecs.open('out.patch', 'w', 'utf8') as f:
        f.write(added_and_removed('sftags.dat', 'nsf_tag.dat'))
