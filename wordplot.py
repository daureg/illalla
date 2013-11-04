#! /usr/bin/python2
# vim: set fileencoding=utf-8


def tag_cloud(words, coords, label):
    from matplotlib.font_manager import FontProperties
    import matplotlib.pyplot as plt
    plt.figure(dpi=200)

    font0 = FontProperties()
    pos = {'horizontalalignment':  'center', 'verticalalignment': 'center'}
    font0.set_size(6)
    # words = ['hello', 'world']
    # coords = [(.8, -3), (-2, 5)]
    plt.plot([p[0] for p in coords], [p[1] for p in coords], 'r.')
    if label:
        for w, c in zip(words, coords):
            plt.text(c[0], c[1], w, fontproperties=font0, **pos)
    plt.axis('off')
    plt.savefig('tag_cloud.pdf', format='pdf', transparent=True)
