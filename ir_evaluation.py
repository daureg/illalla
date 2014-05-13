from collections import defaultdict
import utils as u
import numpy as np


def count_categories(raw_categories):
    """Count the venues categories given by `raw_categories` and returns:
    * sub_count: {sub category index: number of venues}
    * top_count: {sub category index: total number of venues in the same top
                  category}
    * sub_cat_to_top: {sub category index: corresponding top category index}
    ."""
    top_cat_to_sub = defaultdict(set)
    sub_count = defaultdict(int)
    sub_cat_to_top = {}
    for cat in raw_categories:
        sub_count[cat] += 1
        top_cat_id = 1000 * (cat/1000)
        sub_cat_to_top[cat] = top_cat_id
        top_cat_to_sub[top_cat_id] |= set([cat])
    top_count = {}
    for sub_cats in top_cat_to_sub.itervalues():
        total = sum([sub_count[sub_cat] for sub_cat in sub_cats])
        for sub_cat in sub_cats:
            top_count[sub_cat] = total - sub_count[sub_cat]
    return sub_count, top_count, sub_cat_to_top


def NDCG(gold_cat, results, cats_count, rank):
    """Compute the Normalized Discounted Cumulative Gain at rank K of
    `results`, a ranked list of sub categories, given that we were trying to
    retrieve `gold_cat` among `cats_count`."""
    sub_count, top_count, sub_cat_to_top = cats_count
    coeff = np.log2(np.arange(2, rank+2))

    @u.memodict
    def perfect_score(sub_cat):
        """Compute the maximum score if categories are ordered optimally with
        respect to `sub_cat`."""
        different_cat = max(0, rank - sub_count[sub_cat] - top_count[sub_cat])
        # 2**.4-1 = 0.3195079107728942
        scores = np.array(sub_count[sub_cat]*[1.0, ] +
                          top_count[sub_cat]*[0.3195079, ] +
                          different_cat*[0.0, ])
        return np.sum(scores[:rank] / coeff)

    def relevance(query_cat, result_cat):
        """Compute R, the relevance of `result_cat` with respect to `query_cat`
        and returns 2**R - 1"""
        # if result in brand(query) return 1 where brand returns brand id in
        # matching city
        if query_cat == result_cat:
            return 1.0
        if sub_cat_to_top[query_cat] == sub_cat_to_top[result_cat]:
            return 0.3195079
        return 0.0

    return np.sum(np.array(map(lambda x: relevance(gold_cat, x),
                               results))/coeff) / perfect_score(gold_cat)


def evaluate_by_NDCG(left, right, matching, all_categories):
    """Query all venues in `left` to and return their DCG score when
    `matching` them in `right`."""
    k = int(left['knn'])
    cats_count = count_categories(all_categories[right['city']])
    res = []
    for venue in left['index']:
        query, _, answers, _, _ = matching(venue, left, right)
        query_cat = all_categories[left['city']][query]
        results_cat = all_categories[right['city']][answers]
        res.append(NDCG(query_cat, results_cat, cats_count, k))
    return np.array(res)
