#! /usr/bin/python2
# vim: set fileencoding=utf-8


def save_var(filename, d):
    import cPickle
    with open(filename, 'w') as f:
        pck = cPickle.Pickler(f)
        pck.dump(d)


def load_var(filename):
    import cPickle
    with open(filename) as f:
        pck = cPickle.Unpickler(f)
        d = pck.load()
    return d
