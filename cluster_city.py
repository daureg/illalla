#! /usr/bin/python2
# vim: set fileencoding=utf-8
import sklearn.cluster as cl
import sklearn.metrics as mt
import ClosestNeighbor as cn
import sklearn.preprocessing as pp
import numpy as np
import prettyplotlib as ppl
from collections import Counter
import VenueFeature as vf
import explore as xp
import CommonMongo as cm


if __name__ == '__main__':
    # pylint: disable=C0103
    import arguments
    args = arguments.city_parser().parse_args()
    city = args.city
    DB, CLIENT = cm.connect_to_db('foursquare', args.host, args.port)

    clusterer = cl.KMeans(3, n_init=5)
    clusterer = cl.MeanShift(min_bin_freq=3, cluster_all=False)
    clusterer = cl.DBSCAN(eps=5, min_samples=8, metric='cityblock')
    clusterer = cl.AffinityPropagation(damping=.55, affinity='euclidean')
    clusterer = cl.SpectralClustering(3, affinity='cosine', n_init=3)

    hel = cn.load_matrix(city)
    features = hel['v']
    scale = pp.MinMaxScaler(copy=False)
    scale.fit_transform(features[:, 0:3])
    scores = []
    for k in range(3, 16):
        clusterer = cl.KMeans(k, n_init=10, tol=1e-5, max_iter=500)
        labels = clusterer.fit_predict(features)
        scores.append(mt.silhouette_score(features, labels))
        print(Counter(labels))
    np.argsort(scores)[::-1]+3
    ppl.plot(range(3, 16), scores[0:], '+')
    clusterer = cl.MeanShift(min_bin_freq=3, cluster_all=False)
    clusterer = cl.KMeans(6, n_init=20, tol=1e-5, max_iter=500)

    visits = xp.get_visits(CLIENT, xp.Entity.venue, city)
    visitors = xp.get_visitors(CLIENT, city)
    density = vf.estimate_density(city)
    c0, _ = vf.venues_info([_ for _ in hel['i'][labels == 0].tolist()
                            if _ in visits],
                           visits, visitors, density, depth=2, tags_freq=False)
    c5, v = vf.venues_info([v for v in hel['i'][labels == 5].tolist()
                            if v in visits],
                           visits, visitors, density, depth=2, tags_freq=False)
    c0.describe()
    c5.describe()
