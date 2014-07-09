#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Read the output of top metrics and print the DCG score of each metric in
each city"""
from __future__ import print_function
from collections import defaultdict
import itertools
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import prettyplotlib as ppl
import numpy as np
import sys
import cities as c
import pandas as pd
from operator import itemgetter

if __name__ == '__main__':
    RES_SIZE = 5
    query_city = sys.argv[1]
    with open('static/cmp_{}.json'.format(query_city)) as gt:
        results = json.load(gt)
    METRICS = sorted(set([str(_['metric'])
                          for _ in results.values()[0].values()[0]]))
    METRICS = ['femd']
    with open('static/ground_truth.json') as gt:
        regions = json.load(gt)
    DISTRICTS = sorted(regions.keys())
    CITIES = sorted(regions.items()[0][1]['gold'].keys())
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
    print('# Querying from {} {{#{}}}\n'.format(c.FULLNAMES[query_city],
                                                query_city))
    nogt = pd.DataFrame(data={"City": map(lambda x: c.FULLNAMES[x[0]], NO_GOLD),
                              "District": map(itemgetter(1), NO_GOLD)})
    # if len(nogt) > 0:
    #     print('There is no ground truth for:\n')
    #     print(nogt.to_html(index=False))
    RES_QUERIES = {(c, d) for c in results.keys() for d in results[c].keys()}
    # NO_RES = sorted(ALL_QUERIES.difference(set(NO_GOLD), RES_QUERIES))
    # nogt = pd.DataFrame(data={"City": map(lambda x: c.FULLNAMES[x[0]], NO_RES),
    #                           "District": map(itemgetter(1), NO_RES)})
    # if len(nogt) > 0:
    #     print('\nNo result for:\n')
    #     print(nogt.to_html(index=False))
    RES_QUERIES.difference_update(NO_GOLD)


def jaccard(s1, s2):
    """Jaccard similarity between two sets."""
    if len(s1) == 0 == len(s2):
        return 1.0
    return 1.0*len(s1.intersection(s2))/len(s1.union(s2))


def relevance(one_result, all_gold):
    return max([jaccard(one_result, g) for g in all_gold])

if __name__ == '__main__':
    discount_factor = np.log2(np.arange(1, RES_SIZE+1) + 1)


def pad_list(short_list, new_size, value=0.0):
    """Fill `short_list` with `value` until it reaches `new_size` length."""
    return short_list + [value, ] * max(0, new_size - len(short_list))


def DCG(relevance):
    relevance = pad_list(relevance, RES_SIZE)
    return np.sum((np.array(relevance)**2 - 1) / discount_factor)

if __name__ == '__main__':
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
            try:
                dst = pad_list(dst, RES_SIZE, max(dst))
            except ValueError:
                print(query_city, city, district, metric)
                dst = RES_SIZE*[0.0, ]
            scores[metric].append(DCG([relevance(r_i, gold)
                                       for r_i in res]) + LOWEST)
            distances[metric].append(dst)
    return scores, distances


def final_result(raw_results):
    scores, distances = compute_scores(raw_results)
    print(query_city, [np.mean(scores[m]) for m in METRICS],
          file=sys.stderr)
    final_order = sorted([(metric, np.mean(dcgs))
                          for metric, dcgs in scores.iteritems()],
                         key=lambda x: -x[1])
    return final_order, scores, distances


if __name__ == '__main___':
    ww, scores, distances = final_result(results)
    nogt = pd.DataFrame(data={"Metric": map(itemgetter(0), ww),
                              "Score": map(itemgetter(1), ww)})
    print('\n## Overall score\n')
    print(nogt.to_html(index=False))

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
    mpl.rcParams['figure.figsize'] = (10, 6.5)
    city_mask = np.array([CITIES.index(city) for city, _ in RES_QUERIES])


    def score_by_cities(scores, order):
        per_city = {metric: [np.mean(np.array(scores[metric])[city_mask == cidx])
                             for cidx in range(len(CITIES))]
                    for metric in METRICS}
        city_names = map(c.FULLNAMES.__getitem__, CITIES)
        mat = pd.DataFrame(index=city_names, data=per_city)
        mat = mat[[_[0] for _ in order]]
        ordered_index = np.array(city_names)[np.argsort(mat.mean(1).values)[::-1]]
        return mat.reindex(pd.Index(ordered_index), copy=False)


    def plot_score(mat):
        ppl.pcolormesh(np.flipud(mat.values), xticklabels=mat.columns.tolist(),
                       yticklabels=mat.index.tolist()[::-1],
                       # , norm=mpl.colors.LogNorm(vmin=mat.values.min(),
                       #                           vmax=mat.values.max())
                       cmap=mpl.cm.YlOrBr)


    mat = score_by_cities(scores, ww)
    print('## Score by city')
    print(mat.to_html())
    plot_score(mat)
    plt.title('Score in {} with no venues weight'.format(c.FULLNAMES[query_city]))
    img_name = 'metrics_from_{}.png'.format(query_city)
    plt.savefig(img_name, dpi=96, transparent=False, frameon=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    print('<img src="{}">\n\n'.format(img_name))
