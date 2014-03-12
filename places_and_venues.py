#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Find a mapping between twitter places id and Foursquare venues id and use it
to fill missing values.

It turns out that some place are not very convenient because they are too
large. For instance, many venues come from tweet with the place
4303d1afc1e98c37, which correspond to the whole city of Moscow. Thus we
restrict ourselves to one to one correspondences, although this may only be
due to the sparsity of the dataset. A better way would be to ask Twitter for
what kind of place it is."""

import CommonMongo as cm


def all_places_from_venue(checkins, city, converse=False):
    """Associate each venue with a list twitter place id (or do the
    `converse`)"""
    query = cm.build_query(city, fields=['lid', 'place'])
    index, values = '$lid', '$place'
    if converse:
        index, values = values, index
    query.append({"$group": {'_id': index, 'others': {'$push': values}}})
    answer = checkins.aggregate(query)['result']
    return {venue['_id']: venue['others'] for venue in answer if venue['_id']}


def build_map(checkins):
    """Associate each twitter place id with a venue when it's not
    ambiguous."""
    common_map = {}
    # do it city by city to not exceed mongo document size limit
    for city in cm.cities.SHORT_KEY:
        res = all_places_from_venue(checkins, city, False)
        venue_to_place = {k: v[0] for k, v in res.iteritems() if len(v) == 1}
        res = all_places_from_venue(checkins, city, True)
        reverse = {v[0]: k for k, v in res.iteritems() if len(v) == 1}
        bijective_venue = set(venue_to_place.keys())
        bijective_venue &= set(reverse.keys())
        common_map.update({v: k for k, v in venue_to_place.iteritems()
                           if k in bijective_venue})
    return common_map


def update_checkins(checkins, cmap):
    """Use the mapping to update venue id of checkins."""
    missing = checkins.find({'lid': None}, {'_id': 1, 'place': 1})
    total, corrected = 0, 0
    for checkin in missing:
        total += 1
        _id, place = checkin['_id'], checkin.get('place', None)
        if place and place in cmap:
            try:
                checkins.update({'_id': _id}, {'$set': {'lid': cmap[place]}})
                corrected += 1
            except cm.pymongo.errors.OperationFailure as err:
                print(err, err.coderr)
    print('correct {} out of {} checkins'.format(corrected, total))

if __name__ == '__main__':
    #pylint: disable=C0103
    import persistent as p
    db = cm.connect_to_db('foursquare')[0]
    # cmap = build_map(db['checkin'])
    # p.save_var('place_to_venue', cmap)
    update_checkins(db['checkin'], p.load_var('place_to_venue'))
