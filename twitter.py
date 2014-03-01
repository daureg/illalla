#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Retrieve checkins tweets"""
import TwitterAPI as twitter
from api_keys import TWITTER_CONSUMER_KEY as consumer_key
from api_keys import TWITTER_CONSUMER_SECRET as consumer_secret
from api_keys import TWITTER_ACCESS_TOKEN as access_token
from api_keys import TWITTER_ACCESS_SECRET as access_secret
import read_foursquare as rf
CITIES_TREE = rf.obtain_tree()
from utils import get_nested
import cities
import locale
locale.setlocale(locale.LC_ALL, 'C')  # to parse date
UTC_DATE = '%a %b %d %X +0000 %Y'


def parse_tweet(tweet):
    """Return a CheckIn from `tweet` or None if it is not located in a valid
    city"""
    # place_type = get_nested(tweet, ['place', 'place_type'])
    # city = None
    # if place_type == 'city':
    #     city = get_nested(tweet, ['place', 'name'])
    loc = get_nested(tweet, 'coordinates')
    if not loc:
        return None
    lon, lat = loc['coordinates']
    city = rf.find_town(lat, lon, CITIES_TREE)
    if city not in cities.SHORT_KEY:
        return None
    id_ = get_nested(tweet, 'id_str')
    urls = get_nested(tweet, ['entities', 'urls'], [])
    # short url of the checkin that need to be expand, either using bitly API
    # or by VenueIdCrawler. Once we get the full URL, we still need to request
    # 4SQ (500 per hours) to get info.
    lid = [url['expanded_url'] for url in urls
           if '4sq.com' in url['expanded_url']][0]
    uid = get_nested(tweet, ['user', 'id_str'])
    try:
        time = rf.datetime.strptime(tweet['created_at'], UTC_DATE)
    except ValueError:
        print(tweet['created_at'])
        time = None
    return rf.CheckIn(id_, lid, uid, city, loc, time)


if __name__ == '__main__':
    #pylint: disable=C0103
    api = twitter.TwitterAPI(consumer_key, consumer_secret,
                             access_token, access_secret)
    r = api.request('statuses/filter', {'track': '4sq com'})
    for item in r.get_iterator():
        print(item)
        break
