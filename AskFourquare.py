#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Parse JSON Foursquare response to relevant python object"""
from datetime import datetime
from read_foursquare import Location
import foursquare
import cities
import CommonMongo as cm
import pycurl
import cStringIO as cs
from utils import get_nested
from lxml import etree
PARSER = etree.HTMLParser()
# thanks Google Chrome (although this is rather fragile)
XPATH_QUERY = '//*[@id="container"]/div/div[2]/div[2]/div[2]/ul'
from collections import namedtuple
from api_keys import FOURSQUARE_ID as CLIENT_ID
from api_keys import FOURSQUARE_SECRET as CLIENT_SECRET
from twitter_helper import obtain_tree, find_town
CITIES_TREE = obtain_tree()

# https://developer.foursquare.com/docs/responses/venue
Venue = namedtuple('Venue', ['id', 'name', 'loc', 'cats', 'cat',
                             'checkinsCount', 'usersCount', 'tipCount',
                             'hours', 'price', 'rating', 'createdAt', 'mayor',
                             'tags', 'shortUrl', 'canonicalUrl', 'likes',
                             'likers', 'city', 'closed'])
# https://developer.foursquare.com/docs/responses/user
User = namedtuple('User', ['id', 'firstName', 'lastName', 'friends',
                           'friendsCount', 'gender', 'homeCity', 'tips',
                           'lists', 'badges', 'mayorships', 'photos',
                           'checkins'])


def parse_opening_time(info):
    """ the relevant fields of info look something like this:
        {"timeframes":[
            {"days":"Mon–Wed, Fri",
            "open":[
            {"renderedTime":"5:30 AM–Noon",
             "renderedTime":"2:45 PM–Midnight",
            }],
            },
            {"days":"Thu, Sat–Sun",
            "open":[
            { "renderedTime":"8:00 AM–9:00 PM"
            }],
            }]
        }
        A machine readable version is available, but it would costs another
        call: https://developer.foursquare.com/docs/responses/hours.html
        Beyond parsing, the issue is that I don't how to represent that more
        compactly: maybe by doing some simplification.
    """
    return None


def venue_profile(venue):
    """Return a Venue object from a venue json description."""
    assert len(venue.keys()) > 1, "don't send top-level object"
    vid = venue['id']
    name = venue['name']
    loc = venue['location']
    try:
        lon, lat = loc['lng'], loc['lat']
        loc = Location('Point', [lon, lat])._asdict()
        city = find_town(lat, lon, CITIES_TREE)
    except KeyError:
        print(vid, loc)
        # Because loc is 2dsphere index, one cannot insert document with no
        # location. I could have use 2d index (because I will use only flat
        # geometry but then there are limitation on compound index:
        # http://docs.mongodb.org/manual/applications/geospatial-indexes/
        return None
    # if city is None:
    #     print("can't match {}".format(venue['location']))
    cats = [c['id'] for c in venue['categories']]
    cat = None if len(cats) == 0 else cats.pop(0)
    checkinsCount = venue['stats']['checkinsCount']
    usersCount = venue['stats']['usersCount']
    tipCount = venue['stats']['tipCount']
    hours = None
    if 'hours' in venue:
        hours = parse_opening_time(venue['hours'])
    price = None if 'price' not in venue else venue['price']['tier']
    rating = None if 'rating' not in venue else venue['rating']
    createdAt = datetime.fromtimestamp(venue['createdAt'])
    mayor = None
    if 'mayor' in venue and 'user' in venue['mayor']:
        mayor = int(venue['mayor']['user']['id'])
    tags = list(set([t.strip() for t in venue['tags']]))
    shortUrl = venue['shortUrl']
    canonicalUrl = venue['canonicalUrl']
    likers, likes = get_list_of('likes', venue)
    closed = None if 'closed' not in venue else venue['closed']

    return Venue(vid, name, loc, cats, cat, checkinsCount, usersCount,
                 tipCount, hours, price, rating, createdAt, mayor, tags,
                 shortUrl, canonicalUrl, likes, likers, city, closed)


def user_profile(user):
    """Return a User object from a user json description."""
    assert len(user.keys()) > 1, "don't send top-level object"
    uid = int(user['id'])
    firstName = user.get('firstName', None)
    lastName = user.get('lastName', None)
    if not firstName or not lastName:
        print('missing info for {}'.format(uid))
    # only a sample of 10 friends, to get them all, call
    # https://developer.foursquare.com/docs/users/friends.html
    friends, friendsCount = get_list_of('friends', user)
    gender = None if user['gender'] == "none" else user['gender']
    homeCity = None if user['homeCity'] == "" else user['homeCity']
    tips = get_count(user, 'tips')
    lists = get_count(user, 'lists')
    badges = get_count(user, 'badges')
    mayorships = get_count(user, 'mayorships')
    photos = get_count(user, 'photos')
    checkins = get_count(user, 'checkins')

    return User(uid, firstName, lastName, friends, friendsCount, gender,
                homeCity, tips, lists, badges, mayorships, photos, checkins)


def get_count(obj, field):
    """If available, return how many item of type 'field' are in 'obj'"""
    return get_nested(obj, [field, 'count'], 0)


def get_list_of(field, obj):
    """Return a list of id of item of type 'field' within 'obj'"""
    count = get_count(obj, field)
    if count > 0 and 'groups' in obj[field]:
        groups = [g['items'] for g in obj[field]['groups']
                  if g['type'] == 'others']
        return [int(u['id']) for g in groups for u in g], count
    return [], 0


def gather_all_entities_id(checkins, entity='lid', city=None, limit=None):
    """Return at most `limit` `entity` ids within `city`."""
    assert entity in ['lid', 'uid'], 'what are you looking for?'
    query = [
        {'$match': {'lid': {'$ne': None}}},
        {'$project': {'_id': 0, entity: 1}},
        {'$group': {'_id': '$'+entity, 'count': {'$sum': 1}}},
    ]
    if isinstance(city, str) and city in cities.SHORT_KEY:
        query[0]['$match']['city'] = city
    if isinstance(limit, int) and limit > 0:
        query.extend([{'$sort': {'count': -1}}, {'$limit': limit}])
    res = checkins.aggregate(query)['result']
    print('{} matched'.format(len(res)))
    return [e['_id'] for e in res]


def similar_venues(vid, venue_db=None, client=None):
    """Return similar venues to `vid` as suggested by Foursquare, either by
    scrapping webpage (at most 3 results) or making an API call using `client`
    (at most 5 results)."""
    assert venue_db or client, 'Need more information'
    venue = venue_db.find_one({'_id': vid},
                              {'canonicalUrl': 1, 'similars': 1})
    if not venue:
        print(vid + ' is not an existing venue id')
        return None
    if 'similars' in venue and not client:
        # if we have a client and the venue was previously just crawled, we
        # may find more similar venues (4 and 5)
        return venue['similars']
    if client:
        try:
            answer = client.venues.similar(vid)['similarVenues']['items']
            return [str(v['id']) for v in answer]
        except foursquare.FoursquareException as e:
            print(e)
            return None

    buf = cs.StringIO()
    fetcher = pycurl.Curl()
    fetcher.setopt(pycurl.URL, answer['canonicalUrl'])
    fetcher.setopt(pycurl.WRITEFUNCTION, buf.write)
    try:
        fetcher.perform()
    except pycurl.error as e:
        print(str(e))
        return None
    page = buf.getvalue()
    tree = etree.fromstring(page, PARSER)
    similars = tree.xpath(XPATH_QUERY)
    if len(similars) == 0:
        return []
    return [c.attrib['data-venueid'] for c in similars[0].iterchildren()]

if __name__ == '__main__':
    client = foursquare.Foursquare(CLIENT_ID, CLIENT_SECRET)
    import arguments
    args = arguments.city_parser().parse_args()
    db = cm.connect_to_db('foursquare', args.host, args.port)[0]
    checkins = db['checkin']
    # print(venue_profile(client, ''))
    # up = user_profile(client, 2355635)
    # vids = ['4a2705e6f964a52048891fe3', '4b4ad9dff964a5200b8f26e3',
    #         '40a55d80f964a52020f31ee3', '4b4ad9dff964c5200']
    # [client.venues(vid, multi=True) for vid in vids]
    # answers = list(client.multi())
    r = gather_all_entities_id(checkins, city='helsinki', limit=50)
    for b in r:
        print(b)
    # svids = ['4c4787646c379521a121cfb5', '43222200f964a5209a271fe3',
    #          '4b218c2ef964a520a83d24e3']
    # gold = [[], ['4bbd0fbb8ec3d13acea01b28', '451d2412f964a5208a3a1fe3'],
    #         ['4d72a2a9ec07548190588cbf', '4a736a23f964a52062dc1fe3',
    #          '4f2d3e99e4b056f83aecdc88', '4aa7b5e4f964a520064d20e3',
    #          '4b2a2b72f964a520bca524e3']]
    # for query, response in zip(svids, gold):
    #     print(similar_venues(query, client=client), response)
