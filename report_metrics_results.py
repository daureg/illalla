#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Read the output of top metrics and print the DCG score of each metric in
each city"""
from collections import defaultdict
import itertools
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import prettyplotlib as ppl
import numpy as np
import sys

RES_SIZE = 5
query_city = sys.argv[1]
with open('static/cmp_{}.json'.format(query_city)) as gt:
    results = json.load(gt)
METRICS = set([str(_['metric']) for _ in results.values()[0].values()[0]])
with open('static/ground_truth.json') as gt:
    regions = json.load(gt)
DISTRICTS = regions.keys()
CITIES = regions.items()[0][1]['gold'].keys()
assert query_city in CITIES, ', '.join(CITIES)
CITIES.remove(query_city)
ALL_QUERIES = set(itertools.product(CITIES, DISTRICTS))
NO_GOLD = []
for d, g in regions.iteritems():
    g = g['gold']
    missing = set(CITIES).difference(set(g.keys()))
    for city in missing:
        NO_GOLD.append((city, d))
print_region = lambda x: '  • '+str(x[0]).ljust(15)+str(x[1])
print('The query city is {}.'.format(query_city))
print('No ground truth for:')
print('\n'.join(map(print_region, NO_GOLD)))
RES_QUERIES = {(c, d) for c in results.keys() for d in results[c].keys()}
print('No result for:')
print('\n'.join(map(print_region,
                    ALL_QUERIES.difference(set(NO_GOLD), RES_QUERIES))))
RES_QUERIES.difference_update(NO_GOLD)


def jaccard(s1, s2):
    """Jaccard similarity between two sets."""
    if len(s1) == 0 == len(s2):
        return 1.0
    return 1.0*len(s1.intersection(s2))/len(s1.union(s2))


def relevance(one_result, all_gold):
    return max([jaccard(one_result, g) for g in all_gold])


discount_factor = np.log2(np.arange(1, RES_SIZE+1) + 1)


def pad_list(short_list, new_size, value=0.0):
    """Fill `short_list` with `value` until it reaches `new_size` length."""
    return short_list + [value, ] * max(0, new_size - len(short_list))


def DCG(relevance):
    relevance = pad_list(relevance, RES_SIZE)
    return np.sum((np.array(relevance)**2 - 1) / discount_factor)


LOWEST = -DCG(RES_SIZE*[0.0, ])



def compute_scores(raw_result):
    scores = defaultdict(list)
    distances = defaultdict(list)
    for city, district in RES_QUERIES:
        gold = [set(_['properties']['venues'])
                for _ in regions[district]['gold'][city]]
        for metric in METRICS:
            res = [set(_['venues']) for _ in raw_result[city][district]
                   if _['metric'] == metric]
            dst = [_['dst'] for _ in raw_result[city][district]
                   if _['metric'] == metric]
            dst = pad_list(dst, RES_SIZE, max(dst))
            scores[metric].append(DCG([relevance(r_i, gold)
                                       for r_i in res]) + LOWEST)
            distances[metric].append(dst)
    return scores, distances


def final_result(raw_results):
    scores, distances = compute_scores(raw_results)
    final_order = sorted([(metric, np.mean(dcgs))
                          for metric, dcgs in scores.iteritems()],
                         key=lambda x: -x[1])
    print('Overall metrics score:')
    for met, scr in final_order:
        print('{}{:.5f}'.format(met.ljust(19), scr))
    return final_order, scores, distances
ww, scores, distances = final_result(results)

# Plot of individual query:
# QUERIES = [city[:3]+'_'+district[:5]
#            for city, district in itertools.product(CITIES, DISTRICTS)
#            if (city, district) not in MISSING]
# lines = []
# plt.xticks(range(len(QUERIES)), QUERIES, rotation=45)
# for idx, dcgs in enumerate(scores.itervalues()):
#     lines.extend(ppl.plot(dcgs, lw=1.25, c=ppl.colors.set1[idx]))
# _=plt.legend(lines, scores.keys())
# _=plt.xlabel('query')
# _=plt.ylabel('score')

# There is no obvious correlation between the average raw distance of each
# result and the score of the corresponding metric
# for idx, metric in enumerate(METRICS):
#     # ppl.plot(scores[metric], np.mean(distances[metric], 1), '.', ms=6,
#     #          c=ppl.colors.set1[idx])
#     _ = ppl.scatter(scores[metric], np.mean(distances[metric], 1))
# _ = plt.xlabel('score')
# _ = plt.ylabel('mean distance')

# We can also break the scores by cities
import pandas as pd
mpl.rcParams['figure.figsize'] = (12, 8)
city_mask = np.array([CITIES.index(city) for city, _ in RES_QUERIES])


def score_by_cities(scores, order):
    per_city = {metric: [np.mean(np.array(scores[metric])[city_mask == cidx])
                         for cidx in range(len(CITIES))]
                for metric in METRICS}
    mat = pd.DataFrame(index=CITIES, data=per_city)
    mat = mat[[_[0] for _ in order]]
    ordered_index = np.array(CITIES)[np.argsort(mat.mean(1).values)[::-1]]
    return mat.reindex(pd.Index(ordered_index), copy=False)


def plot_score(mat):
    ppl.pcolormesh(np.flipud(mat.values), xticklabels=mat.columns.tolist(),
                   yticklabels=mat.index.tolist()[::-1],
                   # , norm=mpl.colors.LogNorm(vmin=mat.values.min(),
                   #                           vmax=mat.values.max())
                   cmap=mpl.cm.YlOrBr)


mat = score_by_cities(scores, ww)
print('Score by city')
print(mat)
plot_score(mat)
plt.title('Score in {} with no venues weight'.format(query_city))
plt.savefig('metrics_from_{}'.format(query_city), dpi=96, transparent=False,
            frameon=False, bbox_inches='tight', pad_inches=0.1)
plt.clf()
