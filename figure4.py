#! /usr/bin/python2
# vim: set fileencoding=utf-8
import matplotlib as mpl
mpl.use('pgf')

pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.serif": [],                   # use latex default serif font
    "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
    "text.usetex": True,    # use inline math for ticks
    "pgf.rcfonts": False,   # don't setup fonts from rc parameters
    "pgf.texsystem": "lualatex",
    "pgf.preamble": [
        r"\usepackage[T1]{fontenc}",
    ]
}
mpl.rcParams.update(pgf_with_rc_fonts)

import matplotlib.pyplot as plt
import numpy as np
import prettyplotlib as ppl
import ujson
import neighborhood as nb
from scipy.spatial.distance import cdist

# load data
with open('static/ground_truth.json') as infile:
    gold_list = ujson.load(infile)
districts = sorted(gold_list.iterkeys())
cities = sorted(gold_list[districts[0]]['gold'].keys())
euclidean_ground_metric = {name: nb.cn.gather_info(name, raw_features=True, hide_category=True) 
                           for name in cities}
lmnn_ground_metric = {name: nb.cn.gather_info(name, raw_features=False, hide_category=True) 
                      for name in cities}
cities_desc = euclidean_ground_metric

query_city = 'paris'
district = 'pigalle'
target_city = 'barcelona'

query_venues = gold_list[district]['gold'][query_city][0]['properties']['venues']
mask = np.where(np.in1d(cities_desc[query_city]['index'], query_venues))[0]
query_features = cities_desc[query_city]['features'][mask, :]
all_target_features = cities_desc[target_city]['features']

print('Venues in query: {}'.format(len(query_venues)))
print('Venues in target cities: {}'.format(len(all_target_features)))
distances = cdist(query_features, all_target_features)

gold_target_venues_indices = [np.where(np.in1d(cities_desc[target_city]['index'], reg['properties']['venues']))[0] 
                              for reg in gold_list[district]['gold'][target_city]]
print('There are {} corresponding ground truth areas'.format(len(gold_target_venues_indices)))

sorted_distances = np.sort(distances, 1)
ordered = np.argsort(distances, 1)
query_order=np.argsort(sorted_distances[:, 50])

fig, ax = plt.subplots(1)
_=ppl.pcolormesh(fig, ax, sorted_distances[query_order, :], norm=mpl.colors.LogNorm(vmin=sorted_distances[:, 0].min(), vmax=sorted_distances[:, -1].max()))
tg=0
for qv in query_order:
    venue_indices_sorted = np.argsort(ordered[qv, :])
    _=ppl.plot(venue_indices_sorted[gold_target_venues_indices[tg]], qv*np.ones(len(gold_target_venues_indices[tg]))+.5, 'kx', ms=6, mew=1.3)
fs=12
plt.ylabel('Venues in query region', fontsize=fs)
# plt.xlabel('Venues in {}, sorted on each row by distance with the corresponding query venue'.format(target_city.title()), fontsize=16)
plt.xlabel('Venues in {}'.format(target_city.title()), fontsize=fs)
title = '{}, from {} to {}\nThe crosses mark position of venues in ground truth region\nThe color encodes distance between venues, the more red the farthest'
#plt.title(title.format(district.title(), query_city.title(), target_city.title()))
_=plt.xlim([0, len(all_target_features)])
ax.tick_params(axis='both', which='major', labelsize=fs)
fig.delaxes(fig.axes[1])
ax.set_aspect(45)
plt.savefig('fig4.pgf')
