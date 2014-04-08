#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Use feature vector city matrix to perform one to one venue query."""
import os
import numpy as np
import sklearn.neighbors as skn
import arguments
# import CommonMongo as cm
import VenueFeature as vf
import pandas as pd

FEATURES = ['likes', 'users', 'checkins', 'publlicness', 'density',
            'category', 'art', 'education', 'food', 'night', 'recreation',
            'shop', 'professional', 'residence', 'transport', 'focus',
            'photogenicity', 'weekend']
for i in range(6, 15):
    FEATURES[i] += ' surrounding'
FEATURES.extend(['activity at ' + t for t in vf.named_ticks('day', 1, 4)])


def load_matrix(city):
    """Open `city` matrix or compute it."""
    filename = city + '_fv.mat'
    if not os.path.exists(filename):
        vf.describe_city(city)
    mat = vf.sio.loadmat(filename)
    # TODO: do not generate them in the first place?
    # pylint: disable=E1101
    # TODO: zscore everything
    mat['v'][np.isinf(mat['v'])] = 1e9
    return mat


def gather_info(city):
    """Build KD-tree for each categories of venues in `city`."""
    matrix = load_matrix(city)
    res = {'features': matrix['v']}
    for cat in range(len(vf.CATS)):
        cat *= 1e5
        mask = res['features'][:, 5] == cat
        venues = matrix['i'][mask]
        res.update({int(cat):
                    (skn.NearestNeighbors(1).fit(res['features'][mask, :]),
                     venues)})
    res.update({'index': list(matrix['i'].ravel())})
    return res


def find_closest(vid, origin, dest):
    """Find the closest venue in `dest` to `vid`, which lies in `origin`."""
    try:
        query = origin['index'].index(vid)
    except ValueError:
        return None, None, None
    venue = origin['features'][query, :]
    cat = int(venue[5])
    closest_idx = dest[cat][0].kneighbors(venue)[1].ravel()[0]
    print(cat, closest_idx)
    res_id = dest[cat][1][closest_idx].ravel()[0]
    answer = dest['index'].index(res_id)
    return query, res_id, answer


def interpret(query, answer):
    """Return a sorted of list of criteria explaining distance between `query`
    and `answer`."""
    diff = (query - answer) * (query - answer)
    # pylint: disable=E1101
    smaller_first = np.argsort(diff)
    percentage = 100*diff/np.sum(diff)
    return [{'query': query[f], 'answer': answer[f], 'feature': FEATURES[f],
             'percentage': percentage[f]} for f in smaller_first]

if __name__ == '__main__':
    # pylint: disable=C0103
    args = arguments.two_cities().parse_args()
    # db, client = cm.connect_to_db('foursquare', args.host, args.port)
    # left = gather_info(args.origin)
    # right = gather_info(args.dest)
    def explain(query, answer):
        columns = 'feature query percentage answer'.split(
        f,q,p,a=vf.u.xzip(interpret(query, answer), columns))
        return pd.DataFrame(data={'feature': f, 'query': q,
                                  'percentage': p, 'answer': a},
                            columns=columns)
    # while True:
    #     venue_id = raw_input('venue: ').strip()
    #     print(find_closest(venue_id, left, right))
