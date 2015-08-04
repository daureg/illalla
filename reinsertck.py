#! /usr/bin/python2
# vim: set fileencoding=utf-8
from operator import itemgetter
from datetime import datetime
import CommonMongo as cm
import csv
import persistent
import pyhash
TO_BE_INSERTED = []
HASHER = pyhash.spooky_128()
VENUE_LOC = persistent.load_var('venue_loc.my')
TRAD = {}
with open('trad.dat', 'r') as f:
    for line in f:
        old, new = line.strip().split(';')
        TRAD[old] = new


def reformat(line_dict):
    vid = line_dict['vid']
    if vid in TRAD:
        vid = TRAD[vid]
    if vid not in VENUE_LOC:
        return None
    if line_dict['_id'] == 'ICWSM':
        txt = ''.join(itemgetter('uid', 'vid', 'time')(line_dict))
        line_dict['_id'] = hex(HASHER(txt))[2:-1]
    line_dict['uid'] = int(line_dict['uid'])
    line_dict['loc'], line_dict['city'] = VENUE_LOC[vid]
    line_dict['time'] = datetime.strptime(line_dict['time'],
                                          '%Y-%m-%dT%H:%M:%SZ')
    line_dict['lid'] = vid
    del line_dict['vid']
    return line_dict


def mongo_insertion(table):
    global TO_BE_INSERTED
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
    from glob import glob
    sources = sorted(glob('all_*.csv'))
    parser = arguments.get_parser()
    args = parser.parse_args()
    print(args)
    db = cm.connect_to_db('foursquare', args.host, args.port)[0]
    TABLE = db['checkin']
    TABLE.ensure_index([('loc', cm.pymongo.GEOSPHERE),
                        ('city', cm.pymongo.ASCENDING)])
    csv.field_size_limit(sys.maxsize)
    total, unmatched = 0, 0
    for fn in sources:
        with open(fn, 'rb') as f:
            reader = csv.DictReader(f, delimiter=';')
            for i, row in enumerate(reader):
                checkin = reformat(row)
                total += 1
                if checkin:
                    TO_BE_INSERTED.append(checkin)
                else:
                    unmatched += 1
                if len(TO_BE_INSERTED) == 400:
                    mongo_insertion(TABLE)
        mongo_insertion(TABLE)
    fail = float(100*unmatched)/total
    print('{} check-ins insered, {:.2f} fail'.format(total-unmatched, fail))
