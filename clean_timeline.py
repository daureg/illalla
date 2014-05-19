#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Separate tweets by cities and remove duplicate."""
from datetime import datetime
import ujson
RE_TIMELINE = r'timeline_[0-9_]+\.json'
RE_TWEETS = r'tweets_2014[0-9_]+\.json'


def load_existing_ids(cmd_args):
    """Read checkins ids in city from disk or DB."""
    city = cmd_args.city
    if city == 'whole':
        return []
    import persistent as p
    try:
        return p.load_var(city+'_checkins_ids.my')
    except IOError:
        pass
    import CommonMongo as cm
    db = cm.connect_to_db('foursquare', cmd_args.host, cmd_args.port)[0]
    ids = {str(_['_id']) for _ in db.checkin.find({'city': city}, {'_id': 1})
           if not isinstance(_['_id'], long)}
    p.save_var(city+'_checkins_ids.my', ids)
    return ids


def load_tweets(directory, city, existing):
    """Reads tweets in `directory` except those already `existing` and return
    a list (with no duplicate) of those in `city` and another list from the
    ones elsewhere."""
    import os
    import re
    do_ntweets = city == 'whole'
    file_pattern = re.compile(RE_TWEETS if do_ntweets else RE_TIMELINE)
    files = [_ for _ in os.listdir(directory) if file_pattern.match(_)]
    local, outside = {}, {}
    for filename in files:
        with open(os.path.join(directory, filename)) as raw_tweets:
            tweets = ujson.load(raw_tweets)
            if len(tweets) < 2:
                continue
            for tweet in tweets:
                if tweet['_id'] in existing:
                    continue
                res = local if tweet['city'] == city else outside
                if tweet['_id'] in res:
                    this_tid = long(tweet['tid'])
                    inplace_tid = long(res[tweet['_id']]['tid'])
                    if this_tid < inplace_tid:
                        res[tweet['_id']] = tweet
                else:
                    res[tweet['_id']] = tweet
    return local.values(), outside.values()


def save_checkins_json(checkins, city):
    """Write `checkins` as one JSON object per line in a file."""
    import codecs
    if len(checkins) == 0:
        return
    local = checkins[0]['city'] == city
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_name = 'timeline_{}{}_{}.json'.format('' if local else 'not_', city,
                                              now)
    to_str = lambda x: ujson.dumps(x, ensure_ascii=False).decode('utf8')
    with codecs.open(out_name, 'w', 'utf8') as output:
        output.write(u'\n'.join([to_str(tweet).replace('\/', '/')
                                 for tweet in checkins]))

if __name__ == '__main__':
    import arguments
    ARGS = arguments.city_parser().parse_args()
    # pylint: disable=C0103
    city = ARGS.city
    existing = load_existing_ids(ARGS)
    directory = 'ntweets' if city == 'whole' else 'tml'
    local, outside = load_tweets(directory, city, existing)
    save_checkins_json(local, city)
    save_checkins_json(outside, city)
