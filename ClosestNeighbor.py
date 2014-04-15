#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Use feature vector city matrix to perform one to one venue query."""
import os
import numpy as np
import sklearn.neighbors as skn
NN = skn.NearestNeighbors
import arguments
import CommonMongo as cm
import VenueFeature as vf
import pandas as pd
import matplotlib.colors as mcolor
import matplotlib as mpl
import scipy.stats as stats

FEATURES = ['likes', 'users', 'checkins', 'publicness', 'density',
            'category', 'art', 'education', 'food', 'night', 'recreation',
            'shop', 'professional', 'residence', 'transport', 'focus',
            'photogenicity', 'weekend']
for i in range(6, 15):
    FEATURES[i] += ' surrounding'
FEATURES.extend(['activity at ' + t for t in vf.named_ticks('day', 1, 4)])
RESTRICTED = np.array(range(18, 23+1))  # pylint: disable=E1101


def load_matrix(city):
    """Open `city` matrix or compute it."""
    filename = city + '_fv.mat'
    if not os.path.exists(filename):
        vf.describe_city(city)
    mat = vf.sio.loadmat(filename)
    # pylint: disable=E1101
    mat['v'][:, 1:3] = np.log(mat['v'][:, 1:3])
    non_categorical = range(24)
    del non_categorical[non_categorical.index(5)]
    del non_categorical[non_categorical.index(17)]
    mat['v'][:, non_categorical] = stats.zscore(mat['v'][:, non_categorical])
    return mat


def gather_info(city, knn=2):
    """Build KD-tree for each categories of venues in `city` that will return
    `knn` results on subsequent call."""
    matrix = load_matrix(city)
    res = {'features': matrix['v']}
    for cat in range(len(vf.CATS)):
        cat *= 1e5
        mask = res['features'][:, 5] == cat
        venues = matrix['i'][mask]
        if len(venues) > 0:
            idx_subset = np.ix_(mask, RESTRICTED)  # pylint: disable=E1101
            res[int(cat)] = (NN(knn).fit(res['features'][idx_subset]), venues)
    res['index'] = list(matrix['i'])
    return res


def find_closest(vid, origin, dest):
    """Find the closest venues in `dest` to `vid`, which lies in `origin`."""
    try:
        query = origin['index'].index(vid)
        venue = origin['features'][query, :]
        cat = int(venue[5])
        venue = venue[RESTRICTED]
        dst, closest_idx = [r.ravel() for r in dest[cat][0].kneighbors(venue)]
    except (ValueError, KeyError) as oops:
        print(oops)
        return None, None, None, None, None
    res_ids = dest[cat][1][closest_idx].ravel()
    answer = [dest['index'].index(rid) for rid in res_ids]
    return query, res_ids, answer, dst, len(dest[cat][1])


def interpret(query, answer, feature_order=None):
    """Return a list of criteria explaining distance between `query`
    and `answer`, along with their value for `answer`. If no `feature_order`
    is provided, one is computed to sort features by the proportion they
    contribute to the total distance."""
    query = query[RESTRICTED]
    answer = answer[RESTRICTED]
    diff = (query - answer) * (query - answer)
    # pylint: disable=E1101
    if feature_order is None:
        feature_order = np.argsort(diff)
    percentage = 100*diff/np.sum(diff)
    colormap = mpl.cm.ScalarMappable(mcolor.Normalize(0, 15), 'copper_r')
    get_color = lambda v: mcolor.rgb2hex(colormap.to_rgba(v))
    sendf = lambda x, p: ('{:.'+str(p)+'f}').format(float(x))
    query_info = [{'val': sendf(query[f], 5),
                   'feature': FEATURES[RESTRICTED[f]]}
                  for f in feature_order]
    answer_info = [{'answer': sendf(answer[f], 5),
                    'percentage': sendf(percentage[f], 4),
                    'color': get_color(float(percentage[f]))}
                   for f in feature_order]
    return query_info, answer_info, feature_order

if __name__ == '__main__':
    # pylint: disable=C0103
    args = arguments.two_cities().parse_args()
    db, client = cm.connect_to_db('foursquare', args.host, args.port)
    left = gather_info(args.origin)
    right = gather_info(args.dest)

    def explain(query, answer):
        columns = 'feature query percentage answer'.split()
        f, q, p, a = vf.u.xzip(interpret(query, answer), columns)
        return pd.DataFrame(data={'feature': f, 'query': q,
                                  'percentage': p, 'answer': a},
                            columns=columns)
