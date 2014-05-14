from scipy.spatial.distance import cdist
from collections import defaultdict, Counter
import utils as u
import numpy as np


def count_categories(raw_categories):
    """Count the venues categories given by `raw_categories` and returns:
    * sub_count: {sub category index: number of venues}
    * top_count: {sub category index: total number of venues in the same top
                  category}
    * sub_cat_to_top: {sub category index: corresponding top category index}
    ."""
    sub_count = defaultdict(int)
    top_cats = range(0, 9*int(1e5), int(1e5))
    sub_cat_to_top = {sub: top for top in top_cats
                      for sub in range(top, top+200)}
    sub_count.update(Counter(raw_categories))
    top_count = defaultdict(int)
    for top_cat in top_cats:
        sub_cats = range(top_cat, top_cat+200)
        total = sum([sub_count[sub_cat] for sub_cat in sub_cats])
        for sub_cat in sub_cats:
            top_count[sub_cat] = total - sub_count[sub_cat]
    return sub_count, top_count, sub_cat_to_top


def NDCG(gold_cat, results, sub_cat_to_top, rank):
    """Compute the Normalized Discounted Cumulative Gain at rank K of
    `results`, a ranked list of sub categories, given that we were trying to
    retrieve `gold_cat` among `cats_count`."""
    coeff = np.log2(np.arange(2, rank+2))

    @u.memodict
    def relevance(result_cat):
        """Compute R, the relevance of `result_cat` with respect to `query_cat`
        and returns 2**R - 1"""
        # if result in brand(query) return 1 where brand returns brand id in
        # matching city
        if gold_cat == result_cat:
            return 1.0
        if sub_cat_to_top[gold_cat] == sub_cat_to_top[result_cat]:
            return 0.3195079
        return 0.0

    return np.sum(np.array(map(relevance, results)) / coeff)


def evaluate_by_NDCG(left, right, matching, all_categories, mat):
    """Query all venues in `left` to and return their DCG score when
    `matching` them in `right`."""
    k = int(left['knn'])
    cats_count = count_categories(all_categories[right['city']])
    sub_count, top_count, sub_cat_to_top = cats_count
    coeff = np.log2(np.arange(2, k+2))

    @u.memodict
    def perfect_score(sub_cat):
        """Compute the maximum score if categories are ordered optimally with
        respect to `sub_cat`."""
        different_cat = max(0, k - sub_count[sub_cat] - top_count[sub_cat])
        # 2**.4-1 = 0.3195079107728942
        scores = np.array(sub_count[sub_cat]*[1.0, ] +
                          top_count[sub_cat]*[0.3195079, ] +
                          different_cat*[0.0, ])
        return np.sum(scores[:k] / coeff)

    res = []
    metric = 'euclidean'
    if mat is not None:
        metric = 'mahalanobis'
        mat = np.linalg.inv(mat)
    dst = cdist(left['features'], right['features'], metric, VI=mat)
    for venue_order, listed in enumerate(np.argsort(dst, axis=1)):
        query_cat = all_categories[left['city']][venue_order]
        results_cat = all_categories[right['city']][listed[:k]]
        res.append(NDCG(query_cat, results_cat, sub_cat_to_top, k) /
                   perfect_score(query_cat))
    return np.array(res)
