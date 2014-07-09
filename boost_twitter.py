#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Gather more tweets by fetching timeline of previously discovered users."""
# Paginate timeline by hand:
# http://dev.twitter.com/docs/working-with-timelines
# With tweepy:
# http://tweepy.readthedocs.org/en/latest/cursor_tutorial.html
# With TwitterAPI:
# http://geduldig.github.io/TwitterAPI/modules/TwitterAPI.TwitterRestPager.html
from api_keys import TWITTER_CONSUMER_KEY as consumer_key
from api_keys import TWITTER_CONSUMER_SECRET as consumer_secret
from api_keys import TWITTER_ACCESS_TOKEN as access_token
from api_keys import TWITTER_ACCESS_SECRET as access_secret
import tweepy
import httplib
from collections import OrderedDict
import twitter_helper as th
from time import sleep
import CheckinAPICrawler as cac
import logging
logging.basicConfig(filename='timeline.log', level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s')
OLD_DATASET_END = th.datetime(2011, 2, 1)
START_OF_TIME = th.datetime(2007, 1, 1)
# mean number of tweets at each hour of Helsinki local summer time + 0.8 std
NB_RESERVED_CALLS = 24*[0, ]


def checkins_from_timeline(napi, user):
    """Return a list of checkins from the last tweets of `user`."""
    # NOTE: replies can also contain checkin it seems
    # https://twitter.com/deniztrkn/status/454328354943299584
    pages = tweepy.Cursor(napi.user_timeline, user_id=user, count='200',
                          trim_user='true', exclude_replies='false',
                          include_rts='false')
    logging.info('retrieving tweets of {}'.format(user))
    res = []
    timeline = pages.items()
    failed_read = 0
    while True:
        try:
            tweet = timeline.next()
            # logging.info('tweet: {}'.format())
        except tweepy.error.TweepError:
            # For instance, 155877671 is not valid anymore
            logging.exception('Issue with {}'.format(user))
            break
        except StopIteration:
            # logging.exception('stop')
            break
        except httplib.IncompleteRead:
            failed_read += 1
            if failed_read >= 5:
                raise
            sleep(25)
            continue
    # for tweet in timeline:
        if not tweet:
            continue
        date = th.datetime.strptime(tweet._json['created_at'], th.UTC_DATE)
        if date < START_OF_TIME:
            break
        parsed = th.parse_tweet(tweet._json)
        if parsed:
            res.append(parsed)
    logging.info('retrieved {} checkins from {}'.format(len(res), user))
    return res


def twitter_rate_info(napi):
    """Return how many timeline requests can still be made before the reset."""
    info = napi.rate_limit_status()['resources']['statuses']
    info = info['/statuses/user_timeline']
    return info['remaining'], info['reset']


def foursquare_rate_info(fs_client):
    """Return approximately how many checkins requests can still be made."""
    from_http_header = int(fs_client.rate_remaining or '5000')
    if from_http_header > 500:
        from_http_header -= 4500
    multi_size = cac.foursquare.MAX_MULTI_REQUESTS
    nb_reserved_calls = NB_RESERVED_CALLS[th.datetime.now().hour]
    return multi_size*from_http_header - nb_reserved_calls


def post_process(batch):
    infos = crawler.checkins_from_url([c.lid for c in batch])
    if not infos or len(infos) == 0:
        return None
    finalized = []
    for checkin, info in zip(batch, infos):
        if info:
            converted = checkin._asdict()
            id_, uid, vid, time = info
            del converted['id']
            # Old checkins could already be in ICWSM dataset yet because of
            # signature, they have a different id. Add a separate sig field
            # for checkins older than that.
            if time.date() < OLD_DATASET_END.date():
                id_, sig = id_.split('?s=')
                converted['sig'] = sig
            converted['_id'] = id_
            converted['uid'] = uid
            converted['lid'] = vid
            converted['time'] = time
            converted['tm'] = True
            finalized.append(converted)
    return finalized


def checkins_from_user(user, napi, crawler, user_types):
    """Save all checkins from `user` and return time to wait before getting
    another one."""
    empty, big, done = user_types
    fetched = []
    checkins = checkins_from_timeline(napi, user)
    if len(checkins) == 0:
        empty.append(user)
    else:
        fs_remaining = foursquare_rate_info(crawler.client)
    if len(checkins) > 1000:
        big.append(user)
        del checkins[:]
    nb_wait = 0
    while len(checkins) > 0 and fs_remaining < len(checkins):
        if nb_wait > 3:
            big.append(user)
            break
        sleep(8*60)
        nb_wait += 1
        batch_size = min(5, len(checkins))
        batch = checkins[:batch_size]
        full_checkins = post_process(batch)
        if full_checkins:
            fetched.extend(full_checkins)
            del checkins[:batch_size]
            fs_remaining = foursquare_rate_info(crawler.client)
    # we have waited enough to grab the remaining checkins in one pass
    full_checkins = post_process(checkins)
    if full_checkins:
        fetched.extend(full_checkins)
        if len(fetched) > 1:
            th.save_checkins_json(fetched, prefix='timeline_'+user)
        done.append(user)
    elif len(checkins) > 0:
        # In that case, we probably got some trouble with Foursquare
        empty.append(user)
    p.save_var('EMPTY.my', empty)
    p.save_var('DONE.my', done)
    p.save_var('BIG.my', big)
    remaining_call, until = twitter_rate_info(napi)
    if remaining_call <= 16 and until > time.time():
        return until + 1 - int(time.time())
    return 0


def get_users(args):
    import CommonMongo as cm
    city = args.city
    try:
        return p.load_var(city+'_users.my')
    except IOError:
        pass
    db = cm.connect_to_db('foursquare', args.host, args.port)[0]
    # First get a list of all users so far
    user_query = cm.build_query(city, venue=True, fields=['tuid'])
    group = {'$group': {'_id': '$tuid', 'checkins': {'$sum': 1}}}
    user_query.extend([group, {'$sort': {'checkins': -1}}])
    users = db.checkin.aggregate(user_query)['result']
    # See how many they are and their check-ins count distribution
    # import utils as u
    # import pandas as pd
    # print(len(users))
    # infos = u.xzip(users, '_id checkins'.split())
    # df_users = pd.DataFrame(index=map(int, infos[0]),
    #                         data=dict(count=infos[1]))
    # ppl.hist(df_users.values, bins=25)
    users = OrderedDict([(_['_id'], _['checkins']) for _ in users])
    return users.keys()

if __name__ == '__main__':
    # import arguments
    import time
    import persistent as p
    import sys
    # ARGS = arguments.city_parser().parse_args()
    ARGS = lambda : None
    ARGS.city = sys.argv[1]
    ARGS.host = 'localhost'
    ARGS.port = 27017
    # pylint: disable=C0103
    users = []

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    napi = tweepy.API(auth, compression=False, wait_on_rate_limit=True,
                      wait_on_rate_limit_notify=True)

    def load_list(name):
        try:
            return p.load_var(name+'.my')
        except IOError:
            return []

    users_id = get_users(ARGS)
    EMPTY, DONE, BIG = load_list('EMPTY'), load_list('DONE'), load_list('BIG')
    # p.save_var(ARGS.city+'_users.my', user_id)
    users_id = set(users_id).difference(EMPTY, DONE, BIG)
    crawler = cac.CheckinAPICrawler()
    # raise Exception('do it yourself!')
    print('Still {} users to process.'.format(len(users_id)))
    import random
    start = time.time()
    end = start + float(sys.argv[2])*60*60
    if len(users_id) == 0:
        with open('nextcity', 'w') as f:
            f.write('sanfrancisco')
    for user in users_id:  # random.sample(users_id, 35):
        print(user)
        time.sleep(checkins_from_user(user, napi, crawler, [EMPTY, BIG, DONE]))
        print(foursquare_rate_info(crawler.client))
        if time.time() > end:
            break
