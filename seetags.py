#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Compare tags of photos and venues."""

if __name__ == '__main__':
    # pylint: disable=C0103
    getvenue = lambda i: db.venue.find_one({'_id': i})
    import CommonMongo as cm
    db, cl = cm.connect_to_db('foursquare')
    flickr = cm.connect_to_db('world', cl)[0]
    res = db.venue.find({'city': 'paris', 'tags': {'$ne': []}}, {'tags': 1})
    venues_tags = {v['_id']: len(v['tags']) for v in res}
    fl_venue = flickr.photos.find({'venue': {'$ne': None}}, {'tags': 1})
    fl_ids = set([v['_id'] for v in fl_venue])
    matching_venue = fl_ids.intersection(venues_tags.keys())
