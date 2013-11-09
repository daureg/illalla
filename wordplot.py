#! /usr/bin/python2
# vim: set fileencoding=utf-8
from random import randint
from utils import to_css_hex
import codecs


def read_entropy(n=200, filename='nentropies.dat'):
    with codecs.open(filename, 'r', 'utf8') as f:
        lines = [i.strip().split() for i in f.readlines()[:n]]
        # see http://stackoverflow.com/a/1747827 for python 2.7+
        tags = dict((tag, float(e)) for e, tag in lines)
    return tags


def tag_cloud(words, coords, label):
    from matplotlib.font_manager import FontProperties
    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    matplotlib.use('PDF')
    import matplotlib.pyplot as plt
    plt.figure(dpi=200)

    H = read_entropy()
    colormap_ = cm.ScalarMappable(colors.Normalize(min(H.values()),
                                                   max(H.values())), 'Blues')
    font0 = FontProperties()
    pos = {'horizontalalignment':  'center', 'verticalalignment': 'center'}
    font0.set_size(1.4)
    for w, c in zip(words, coords):
        plt.plot(c[0], c[1], '.', ms=5, c=to_css_hex(colormap_.to_rgba(H[w])))
        if label:
            plt.text(c[0], c[1], w, fontproperties=font0,
                     rotation=randint(-28, 28), **pos)
    plt.axis('off')
    plt.savefig('red.pdf', format='pdf', transparent=True)


if __name__ == '__main__':
    import more_query as mq
    data = mq.sio.loadmat('MDS1.mat')['A']
    words = mq.get_top_tags(mq.np.size(data, 0), 'nsf_tag.dat')
    coords = [(p[0], p[1]) for p in data]
    tag_cloud(words, coords, True)
