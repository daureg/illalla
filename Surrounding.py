#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Implementation of Surrounding class."""
import sklearn.neighbors as skn
import numpy as np
import utils as u


class Surrounding(object):
    """Build a KD tree of objects to allow faster ball query."""

    def __init__(self, db, query, fields, projection):
        """Retrieve `fields` of items satisfying `query` in `db` and build the
        tree using `projection`."""
        self.fields = fields
        self.id_index_map = {}
        self.loc = []
        self.info = {}
        only_venue_cats = set(['cat', 'cats']) == set(fields)
        if only_venue_cats:  # special case
            self.fields = ['cats']
        missing = 0
        for idx, item in enumerate(db.find(query, {f: 1 for f in fields})):
            id_ = item['_id']
            if id_ not in projection:
                missing += 1
                continue
            self.id_index_map[idx] = id_
            self.loc.append(projection[id_])
            if only_venue_cats:
                self.info[id_] = {'cats': [item['cat']] + item['cats']}
            elif self.fields:
                self.info[id_] = {f: item[f] for f in fields}
        print('missed {}'.format(missing))
        self.space = skn.NearestNeighbors(radius=350.0, algorithm='kd_tree',
                                          leaf_size=35)
        self.space.fit(np.array(self.loc))

    def index_to_id(self, idx):
        """Return the id corresponding to `idx`."""
        return self.id_index_map[idx]

    def all(self):
        """Return info about all objects."""
        return self.idx_to_infos(range(len(self.id_index_map)))

    def around(self, center, radius):
        """Return info about all object at distance `radius` from `center`."""
        neighbors_idx = self.space.radius_neighbors([center], radius, False)[0]
        return self.idx_to_infos(neighbors_idx)

    def idx_to_infos(self, idxs):
        """Return info about object with index in `idxs`."""
        neighbors_ids = [self.index_to_id(idx) for idx in idxs]
        neighbors_locs = [self.loc[idx] for idx in idxs]
        extra = []
        if self.fields:
            extra = u.xzip([self.info[id_] for id_ in neighbors_ids],
                           self.fields)
        return neighbors_ids, extra, neighbors_locs

if __name__ == '__main__':
    # pylint: disable=C0103
    from timeit import default_timer as clock
    import CommonMongo as cm
    import random as r
    import arguments
    args = arguments.city_parser().parse_args()
    city = args.city
    radius = 350
    DB, CLIENT = cm.connect_to_db('foursquare', args.host, args.port)
    import persistent as p
    lvenues = p.load_var(city+'_lvenues.my')
    svenues = Surrounding(DB.venue, {'city': city}, 'cat cats'.split(),
                          lvenues)
    test_ids = r.sample(lvenues.keys(), 35)
    start = clock()
    for vid in test_ids:
        me = DB.venue.find_one({'_id': vid}, {'loc': 1, 'city': 1})
        ball = {'$geometry': me['loc'], '$maxDistance': radius}
        neighbors = DB.venue.find({'city': city, 'loc': {'$near': ball}},
                                  {'cat': 1, 'cats': 1, 'loc': 1})
        vids, vcats = zip(*[(v['_id'], [v['cat']] + v['cats'])
                            for v in neighbors])
    # print((clock() - start)/len(test_ids))
    # start = clock()
    # for vid in test_ids:
        avids, acats = svenues.around(lvenues[vid], radius)
        # Check that we get same cats
        print(all([set(vcats[i]) == set(acats[0][avids.index(id_)])
                   for i, id_ in enumerate(vids) if id_ in avids]))
        vids, avids = set(vids), set(avids)
        # and almost same venue id (except those at the border)
        if not vids == avids:
            print(len(vids), len(vids.difference(avids)),
                  len(avids.difference(vids)))
            for missing in vids.difference(avids):
                print(np.linalg.norm(lvenues[missing] - lvenues[vid]))
    print((clock() - start)/len(test_ids))

    lphotos = p.load_var(city+'_lphotos.my')
    photos = CLIENT.world.photos.find({'hint': city, 'loc': {'$near': ball}},
                                      {'venue': 1, 'taken': 1})
    pids, pvenue, ptime = u.xzip(photos, ['_id', 'venue', 'taken'])
    start = clock()
    sphotos = Surrounding(CLIENT.world.photos, {'hint': city},
                          'venue taken'.split(), lphotos)
    print((clock() - start))
    apids, ainfo = sphotos.around(lvenues[test_ids[0]], radius)

    # lcheckins = p.load_var(city+'_lcheckins.my')
    # scheckins = Surrounding(DB.checkin, {'city': city}, ['time'], lcheckins)
    # checkins = DB.checkin.find({'city': city, 'loc': {'$near': ball}},
    #                            {'time': 1, 'loc': 1})
    # cids, ctime = u.xzip(checkins, ['_id', 'time'])
