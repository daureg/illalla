#! /usr/bin/python2
# vim: set fileencoding=utf-8
import persistent
import numpy as np
u = persistent.load_var('user_status_full')
t = np.array([p[0] for p in u.values() if p[1]])
l = np.array([p[0] for p in u.values() if not p[1]])
photos = sum(t) + sum(l)
print('tourists proportion: {}%'.format(100*len(t)/(len(t) + len(l))))
print("tourists' photos proportion: {}%".format(100*sum(t)/photos))
print('tourist 90 percentile: {}'.format(np.percentile(t, 90)))
print('local 90 percentile: {}'.format(np.percentile(l, 90)))
