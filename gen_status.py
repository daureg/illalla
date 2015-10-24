#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Print statistics about database."""
import prettytable as pt
from datetime import datetime as dt
import CommonMongo as cm
import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US')
except locale.Error:
    locale.setlocale(locale.LC_ALL, '')


def ordered(counts, cities, threshold=10):
    """Return `counts` ordered by cities."""
    as_dict = {v['_id']: v['count'] for v in counts}
    count = [as_dict.get(city, 0) for city in cities]
    count.append(sum(count))
    fmt = lambda v: locale.format('%d', v, grouping=True)
    return [fmt(c) if c > threshold else '' for c in count]


if __name__ == '__main__':
    #pylint: disable=C0103
    import arguments
    args = arguments.get_parser().parse_args()
    foursquare, client = cm.connect_to_db('foursquare', args.host, args.port)
    checkins = foursquare.checkin
    venues = foursquare.venue
    photos = client.world.photos
    newer = dt(2011, 5, 1)
    t = pt.PrettyTable()
    t.junction_char = '|'
    checkin = checkins.aggregate([{'$match': {'time': {'$lt': newer}}},
                                  {'$project': {'city': 1}},
                                  {'$group': {'_id': '$city',
                                              'count': {'$sum': 1}}},
                                  {'$sort': {'count': -1}}])
    located = checkins.aggregate([{'$match': {'lid': {'$ne': None},
                                              'time': {'$lt': newer}}},
                                  {'$project': {'city': 1}},
                                  {'$group': {'_id': '$city',
                                              'count': {'$sum': 1}}}])
    newest = checkins.aggregate([{'$match': {'time': {'$gt': newer}}},
                                 {'$project': {'city': 1}},
                                 {'$group': {'_id': '$city',
                                             'count': {'$sum': 1}}},
                                  {'$sort': {'count': -1}}])
    venue = venues.aggregate([{'$project': {'city': 1}},
                              {'$group': {'_id': '$city',
                                          'count': {'$sum': 1}}}])
    flickr = photos.aggregate([{'$project': {'hint': 1}},
                               {'$group': {'_id': '$hint',
                                           'count': {'$sum': 1}}}])
    order = [ck['_id'] for ck in newest]
    newest = checkins.aggregate([{'$match': {'time': {'$gt': newer}}},
                                 {'$project': {'city': 1}},
                                 {'$group': {'_id': '$city',
                                             'count': {'$sum': 1}}}])
    cities_name = [cm.cities.FULLNAMES[n] for n in order] + ['total']
    t.add_column('city', cities_name, 'l')
    t.add_column('ICWSM checkins', ordered(checkin, order), 'r')
    # t.add_column('ICWSM checkins', ordered(located, order), 'r')
    t.add_column('checkins', ordered(newest, order, 0), 'r')
    t.add_column('venues', ordered(venue, order), 'r')
    t.add_column('photos', ordered(flickr, order), 'r')
    table = str(t)
    line_size = len(table)/(len(order)+4)
    start = table.find('\n')+1
    end = table.find('\n', -int(1.5*line_size))
    with open('status_paper.md', 'w') as status:
        status.write(table[start:end])
