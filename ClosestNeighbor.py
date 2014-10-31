#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Use feature vector city matrix to perform one to one venue query."""
import itertools
import numpy as np
import sklearn.neighbors as skn
NN = skn.NearestNeighbors
import arguments
# import CommonMongo as cm
import VenueFeature as vf
import pandas as pd
import matplotlib.colors as mcolor
import matplotlib as mpl
from scipy.stats import zscore
import random as rd
import persistent as p
import ir_evaluation as ir
from collections import namedtuple
LOOP = namedtuple('Loop', 'path dst size')

FEATURES = ['likes', 'users', 'checkins', 'publicness', 'density',
            'category', 'art', 'education', 'food', 'night', 'recreation',
            'shop', 'professional', 'residence', 'transport', 'focus',
            'photogenicity', 'weekend']
for i in range(6, 15):
    FEATURES[i] += ' surrounding'
FEATURES.extend(['activity at ' + t for t in vf.named_ticks('day', 1, 4)])
FEATURES.append('opening')
FEATURES.extend(['surrounding activity at ' + t
                 for t in vf.named_ticks('day', 1, 4)])
RESTRICTED = np.array(range(len(FEATURES)))  # pylint: disable=E1101
LCATS = {}


def load_matrix(city, hide_category=False):
    """Open `city` matrix or compute it."""
    filename = city
    if not filename.endswith('.mat'):
        filename = city + '_fv.mat'
    mat = vf.sio.loadmat(filename)
    log_nb_users = []
    # pylint: disable=E1101
    if filename.endswith('_fv.mat') or filename.endswith('_tsne.mat'):
        # we loaded the raw features, which need preprocessing
        if filename.endswith('_tsne.mat'):
            mat['v'] = np.insert(mat['v'], 4, values=0, axis=1)
        else:
            mat['v'][:, 0:3] = np.log(mat['v'][:, 0:3])
            is_inf = np.argwhere(np.isinf(mat['v'][:, 0:3]))
            mat['v'][is_inf] = 0.0
            log_nb_users = mat['v'][:, 1].flatten()
        LCATS[city] = np.ceil(mat['v'][:, 5]).astype(int)
        mat['v'][:, 5] = np.divide(LCATS[city], 1000)*1000
        if hide_category:
            mat['v'][:, 5] = np.zeros((mat['v'].shape[0],))
        if filename.endswith('_fv.mat'):
            non_categorical = range(len(FEATURES))
            del non_categorical[non_categorical.index(5)]
            del non_categorical[non_categorical.index(17)]
            weird = np.logical_or(np.isinf(mat['v'][:, 16]),
                                  np.isnan(mat['v'][:, 16])).ravel()
            mat['v'][weird, 16] = 0.0
            mat['v'][:, non_categorical] = zscore(mat['v'][:, non_categorical])
    elif filename.endswith('_embed.mat'):
        # add a blank category feature
        mat['v'] = np.insert(mat['v'], 5, values=0, axis=1)
        # but still get the real one for evaluation purpose
        tmp = vf.sio.loadmat(filename.replace('embed', 'fv').split('/')[-1])
        log_nb_users = np.log(tmp['v'][:, 1]).ravel()
        # LCATS[city] = (tmp['v'][:, 5]).astype(int)
    mat['u'] = log_nb_users
    return mat


def gather_info(city, knn=2, mat=None, raw_features=True, hide_category=False):
    """Build KD-tree for each categories of venues in `city` that will return
    `knn` results when called."""
    if raw_features:
        matrix = load_matrix(city, hide_category)
    else:
        matrix = load_matrix('mLMNN2.5/'+city+'_embed.mat')
    res = {'features': matrix['v'], 'city': city.split('_')[0],
           'users': matrix['u']}
    for cat in range(len(vf.CATS)):
        cat *= 1e5
        mask = res['features'][:, 5] == cat
        venues = matrix['i'][mask]
        if len(venues) > 0:
            frange = np.array(range(len(FEATURES)))
            if city.endswith('_tsne.mat'):
                frange = np.arange(5)
            idx_subset = np.ix_(mask, frange)  # pylint: disable=E1101
            algo = NN(knn) if mat is None else NN(knn, metric='mahalanobis',
                                                  VI=np.linalg.inv(mat))
            res[int(cat)] = (algo.fit(res['features'][idx_subset]), venues)
    res['index'] = list(matrix['i'])
    res['knn'] = knn
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
        print(vid)
        print(oops)
        return None, None, None, None, None
    res_ids = dest[cat][1][closest_idx].ravel()
    answer = [dest['index'].index(rid) for rid in res_ids]
    return query, res_ids, answer, dst, len(dest[cat][1])


def make_loop(vid, origin, dest):
    """Compute the loop originating from `vid` in `origin` and going to
    `dest`."""
    smaller_size = int(1e9)

    def forward(id_):
        """Make one step forward from `id_`."""
        qryc, res_id, ansc, dst, size = find_closest(id_, origin, dest)
        same_cat = LCATS[origin['city']][qryc] == LCATS[dest['city']][ansc[0]]
        return res_id[0], dst[0], size, same_cat

    def backward(id_):
        """Make one step backward from `id_`."""
        qryc, res_id, ansc, dst, size = find_closest(id_, dest, origin)
        same_cat = LCATS[dest['city']][qryc] == LCATS[origin['city']][ansc[0]]
        return res_id[0], dst[0], size, same_cat

    go_forward = True
    loop_closed = False
    loop = LOOP([vid], [0], 0)
    matched_cat = 0
    while not loop_closed:
        vid, dst, size, same_cat = (forward if go_forward else backward)(vid)
        matched_cat += int(same_cat)
        loop_closed = vid in loop.path
        loop.path.append(vid)
        loop.dst.append(dst)
        smaller_size = min(smaller_size, size)
        go_forward = not go_forward
    return loop._replace(size=smaller_size), matched_cat


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
    query_info = [{'val': sendf(query[f], 4),
                   'feature': FEATURES[RESTRICTED[f]]}
                  for f in feature_order]
    answer_info = [{'answer': sendf(answer[f], 4),
                    'percentage': sendf(percentage[f], 3),
                    'color': get_color(float(percentage[f]))}
                   for f in feature_order]
    return query_info, answer_info, feature_order


def loop_over_city(origin, dest):
    """Make some loops in all categories."""
    rd.seed(SEED)
    res = {}
    output = []
    matched_cat = []
    nb_venues = []
    for idx, cat in enumerate(vf.CATS):
        res[idx] = ([], [])
        mask = origin['features'][:, 5] == idx*1e5
        venues = list(itertools.compress(origin['index'], mask))
        chosen = rd.sample(venues, len(venues)/7)
        output.append('{} {} venues'.format(len(chosen), cat))
        for venue in chosen:
            loop, matched = make_loop(venue, origin, dest)
            matched_cat.append(matched)
            nb_venues.append(len(loop.path) - 1)
            # NOTE: Actually, loop.size == len(venues)
            ratio = 1.0*(len(loop.dst) - 3)/(2*loop.size + 1 - 3)
            average = np.mean(loop.dst[1:])  # pylint: disable=E1101
            res[idx][0].append(ratio)
            res[idx][1].append(average)
            output.append('{}:\t{}\t{:.5f}\t{:.5f}'.format(venue,
                                                           len(loop.path)-3,
                                                           ratio, average))
    return res, 100.0*sum(matched_cat)/sum(nb_venues), output


def brand_awareness(brand, src, dst):
    """For all venues of brand `brand` in `src`, return the position of the
    first matching venue of the same brand in `dst`, along with a score
    between 0 (best) and 1 (worst)."""
    res = []
    src_venues = p.load_var('{}_{}.my'.format(src['city'], brand))
    dst_venues = p.load_var('{}_{}.my'.format(dst['city'], brand))
    among = 0
    for venue in src_venues:
        _, ids, _, _, among = find_closest(venue, src, dst)
        ranks = [pos for pos, res_id in enumerate(ids)
                 if res_id in dst_venues]
        res.append((len(dst_venues), among, ranks))
    return res


SEED = 1234
if __name__ == '__main__':
    # pylint: disable=C0103
    brands = ["mcdonald's", 'starbucks']
    args = arguments.two_cities().parse_args()
    # mat, raw, suffix = None, True, ''
    # left = gather_info(args.origin+suffix, 350, mat, raw)
    # right = gather_info(args.dest+suffix, 350, mat, raw)
    # ir.evaluate_by_NDCG(left, right, find_closest, LCATS, mat)
    # raise Exception
    # db, client = cm.connect_to_db('foursquare', args.host, args.port)
    import scipy.io as sio
    # learned = sio.loadmat('allthree_A_30_2.mat')['A']
    learned = sio.loadmat('ITMLall.mat')['A']
    # pylint: disable=E1101
    learned = np.insert(learned, 5, values=0, axis=1)
    learned = np.insert(learned, 5, values=0, axis=0)
    learned[5, 5] = 1.0
    extract = lambda r, i: np.array([one for cats_r in r.itervalues()
                                     for one in cats_r[i]])
    metrics = ['Euclidean', 'Random Diagonal', 'ITML', 'GB-LMNN', '2D t-SNE',
               'Random ordering']
    smetrics = ['euclidean', 'diagonal', 'itml', 'lmnn', 'tsne', 'random']
    br_res = [{brand: [] for brand in brands} for method in metrics]
    res_br = [{brand: [] for brand in brands} for method in metrics]
    for seed in range(88, 89):
        # SEED = seed
        # np.random.seed(SEED)
        random_diag = np.diag((22*np.random.random((1, 31))+0.5).ravel())
        mats = [None, random_diag, learned, None, None, None]
        three = []
        four = []
        for idx, mat in enumerate(mats):
            # print(metrics[idx])
            raw = idx != 3
            if idx == 4:
                suffix = '_tsne.mat'
                RESTRICTED = np.arange(5)
            else:
                suffix = ''
                RESTRICTED = np.arange(len(FEATURES))
            fake = idx == 5
            left = gather_info(args.origin+suffix, knn=540, mat=mat,
                               raw_features=raw, hide_category=True)
            right = gather_info(args.dest+suffix, knn=540, mat=mat,
                                raw_features=raw, hide_category=True)
            three.append(ir.evaluate_by_NDCG(left, right, LCATS, mat, fake))
            # four.append(ir.evaluate_by_NDCG(right, left, LCATS, mat, fake))
            # Evaluation by brand
            # for brand in brands:
            #     p.save_var('eval_brands/{}_{}_{}_{}.my'.format(args.origin,
            #                                                    args.dest,
            #                                                    brand,
            #                                                    smetrics[idx]),
            #                brand_awareness(brand, left, right))
            #     p.save_var('eval_brands/{}_{}_{}_{}.my'.format(args.dest,
            #                                                    args.origin,
            #                                                    brand,
            #                                                    smetrics[idx]),
            #                brand_awareness(brand, right, left))

            #     print(seed, idx, brand)
            #     br_res[idx][brand] = brand_awareness(brand, left, right)
            #     res_br[idx][brand] = brand_awareness(brand, right, left)

            # Evaluation by loopiness
            # numres, match, txtres = loop_over_city(left, right)
            # three.append(numres)
            # print(idx, match)
            # outfile = '{}_{}_{}.res'.format(args.origin, args.dest, idx)
            # with open(outfile, 'w') as f:
            #     f.write('\n'.join(txtres))
        # rnd = extract(three[0], 0)-extract(three[1], 0)
        # itml = extract(three[0], 0)-extract(three[2], 0)
        # lmnn = extract(three[0], 0)-extract(three[3], 0)
        # print(2000*np.sum(lmnn[np.argsort(lmnn)[1:]]) -
        #       2000*np.sum(rnd[np.argsort(rnd)[1:]]))
    import cities as C
    w = np.argwhere(three[1] > 0.8).ravel()
    vindex = set(range(left['features'].shape[0]))
    nw = np.array(list(vindex.difference(list(w))))
    print('\t'.join([C.FULLNAMES[args.origin].ljust(15),
                     C.FULLNAMES[args.dest].ljust(15)] +
                    map(lambda x: '{:.4f}'.format(np.mean(x[nw])), three)))

    def explain(query, answer):
        """Explains distance between `query` and `answer` as a data frame."""
        columns = 'feature query percentage answer'.split()
        f, q, p, a = vf.u.xzip(interpret(query, answer), columns)
        return pd.DataFrame(data={'feature': f, 'query': q,
                                  'percentage': p, 'answer': a},
                            columns=columns)
