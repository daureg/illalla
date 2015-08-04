#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""."""
from datetime import datetime
from cities import SHORT_KEY
import CommonMongo as cm
import csv
import persistent
TO_BE_INSERTED = []
VENUE_LOC = {}

def reformat(line_dict):
    global VENUE_LOC
    for field in ['rating', 'checkinsCount', 'likes', 'usersCount',
                  'tipCount', 'price', 'mayor']:
        val = line_dict[field]
        conv = float if field == 'rating' else int
        line_dict[field] = None if len(val) == 0 else conv(line_dict[field])
    val = line_dict['likers']
    line_dict['likers'] = [] if len(val) == 0 else [int(_) for _ in val.split(',')]
    for field in ['tags', 'cats']:
        val = line_dict['tags']
        if len(val) == 0:
            line_dict[field] = []
        else:
            line_dict[field] = line_dict[field].split(',')
    closed = line_dict['closed']
    line_dict['closed'] = None if len(closed) else bool(closed)
    lon, lat = [float(_) for _ in line_dict['lon,lat'].split(',')]
    del line_dict['lon,lat']
    line_dict['loc'] = {'type': 'Point', 'coordinates': [lon, lat]}
    created = line_dict['createdAt']
    line_dict['createdAt'] = datetime.strptime(created, '%Y-%m-%dT%H:%M:%SZ')
    line_dict['_id'] = line_dict['vid']
    del line_dict['vid']
    return line_dict


def mongo_insertion(table):
    global TO_BE_INSERTED
    # return None
    if len(TO_BE_INSERTED) == 0:
        return
    try:
        table.insert(TO_BE_INSERTED, continue_on_error=True)
    except cm.pymongo.errors.DuplicateKeyError:
        pass
    except cm.pymongo.errors.OperationFailure as e:
        print(e, e.code)
    del TO_BE_INSERTED[:]


if __name__ == '__main__':
    # pylint: disable=C0103
    import sys
    import arguments
    parser = arguments.get_parser()
    parser.add_argument('csvin', help='input file')
    parser.add_argument('skip', type=int, default=0, nargs='?',
                        help='number of lines to skip')
    args = parser.parse_args()
    print(args)
    db = cm.connect_to_db('foursquare', args.host, args.port)[0]
    TABLE = db['venue']
    TABLE.ensure_index([('loc', cm.pymongo.GEOSPHERE),
                        ('city', cm.pymongo.ASCENDING),
                        ('cat', cm.pymongo.ASCENDING)])
    filename = sys.argv[1]
    if len(sys.argv) > 2:
        skip = int(sys.argv[2])
    else:
        skip = 0
    csv.field_size_limit(sys.maxsize)
    allowed_cities = set(SHORT_KEY)
    with open(filename, 'rb') as f:
        reader = csv.DictReader(f, delimiter=';', quoting=csv.QUOTE_NONE)
        for i, row in enumerate(reader):
            # if i > 31000:
            #     break
            if i < skip:
                continue
            if row['lon,lat'] is None:
                print(row['vid'])
                continue
            venue = reformat(row)
            # if i == 29950:
            #     print(venue)
            #     break
            # if venue['_id'] == '4ac518c5f964a520c1a420e3':
            #     print(venue, venue['city'] in allowed_cities)
            if venue['city'] in allowed_cities:
                VENUE_LOC[venue['_id']] = (venue['loc'], venue['city'])
                TO_BE_INSERTED.append(venue)
            if len(TO_BE_INSERTED) == 400:
                mongo_insertion(TABLE)
    mongo_insertion(TABLE)
    persistent.save_var('venue_loc.my', VENUE_LOC)
