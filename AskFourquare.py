#! /usr/bin/python2
# vim: set fileencoding=utf-8
from datetime import datetime
from read_foursquare import Location
import foursquare
import cities
import pymongo
from collections import namedtuple
from api_keys import FOURSQUARE_ID as CLIENT_ID
from api_keys import FOURSQUARE_SECRET as CLIENT_SECRET
from persistent import load_var, save_var
from read_foursquare import obtain_tree, find_town
CITIES_TREE = obtain_tree()

# https://developer.foursquare.com/docs/responses/venue
Venue = namedtuple('Venue', ['id', 'name', 'loc', 'cats', 'cat', 'stats',
                             'hours', 'price', 'rating', 'createdAt', 'mayor',
                             'tags', 'shortUrl', 'canonicalUrl', 'likes',
                             'likers', 'city'])
Categories = namedtuple('Categories', ['id', 'name', 'sub'])
# https://developer.foursquare.com/docs/responses/user
User = namedtuple('User', ['id', 'firstName', 'lastName', 'friends',
                           'friendsCount', 'gender', 'homeCity', 'tips',
                           'lists', 'badges', 'mayorships', 'photos',
                           'checkins'])


def parse_categories(top_list):
    if len(top_list) == 0:
        return []
    res = []
    for c in top_list:
        subs = []
        if isinstance(c, dict) and 'categories' in c:
            subs = parse_categories(c['categories'])
        res.append(Categories(c['id'], c['shortName'], subs))
    return res


def get_categories(client=None):
    """Return categories list from disk or from Foursquare website using
    client"""
    if client is None:
        return load_var('categories')['categories']
    raw_cats = client.categories()['categories']
    cats = Categories('0', '_', parse_categories(raw_cats))
    save_var('categories', cats)
    return cats


def search_categories(cats, query, field=None):
    """Return a category matching query (either name or id) and its path inside
    cats."""
    if field is None:
        field = 0 if '4' in query else 1
    if cats[field] == query:
        return cats, [query]
    for c in cats.sub:
        found, path = search_categories(c, query, field)
        if found is not None:
            return found, [cats[field]] + path
    return None, None


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
    lon, lat = loc['lng'], loc['lat']
    loc = Location('Point', [lon, lat])._asdict()
    city = find_town(lat, lon, CITIES_TREE)
    if city is None:
        print(city, lon, lat)
    cats = [c['id'] for c in venue['categories']]
    cat = None if len(cats) == 0 else cats.pop(0)
    stats = [venue['stats'][key]
             for key in ['checkinsCount', 'usersCount', 'tipCount']]
    hours = None
    if 'hours' in venue:
        hours = parse_opening_time(venue['hours'])
    price = None if 'price' not in venue else venue['price']['tier']
    rating = None if 'rating' not in venue else venue['rating']
    createdAt = datetime.fromtimestamp(venue['createdAt'])
    mayor = None
    if 'user' in venue['mayor']:
        mayor = int(venue['mayor']['user']['id'])
    tags = venue['tags']
    shortUrl = venue['shortUrl']
    canonicalUrl = venue['canonicalUrl']
    likers, likes = get_list_of('likes', venue)

    return Venue(vid, name, loc, cats, cat, stats, hours, price, rating,
                 createdAt, mayor, tags, shortUrl, canonicalUrl, likes, likers,
                 city)


def user_profile(user):
    """Return a User object from a user json description."""
    assert len(user.keys()) > 1, "don't send top-level object"
    uid = int(user['id'])
    firstName = user['firstName']
    lastName = user['lastName']
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
    if field in obj and 'count' in obj[field]:
        return obj[field]['count']
    return 0


def get_list_of(field, obj):
    """Return a list of id of item of type 'field' within 'obj'"""
    count = get_count(obj, field)
    if count > 0 and 'groups' in obj[field]:
        groups = [g['items'] for g in obj[field]['groups']
                  if g['type'] == 'others']
        return [int(u['id']) for g in groups for u in g], count
    return [], 0


def gather_all_entities_id(checkins, entity='lid', city=None, limit=None,
                           chunk_size=foursquare.MAX_MULTI_REQUESTS):
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
    batch = []
    for obj in res:
        _id = obj['_id']
        if len(batch) < chunk_size:
            batch.append(_id)
        else:
            yield batch
            batch = [_id]
    yield batch

if __name__ == '__main__':
    client = foursquare.Foursquare(CLIENT_ID, CLIENT_SECRET)
    mongo_client = pymongo.MongoClient('localhost', 27017)
    db = mongo_client['foursquare']
    checkins = db['checkin']
    # print(venue_profile(client, ''))
    # ft = get_categories()
    # up = user_profile(client, 2355635)
    # vids = ['4a2705e6f964a52048891fe3', '4b4ad9dff964a5200b8f26e3',
    #         '40a55d80f964a52020f31ee3', '4b4ad9dff964c5200']
    # [client.venues(vid, multi=True) for vid in vids]
    # answers = list(client.multi())
    r = gather_all_entities_id(checkins, city='helsinki', limit=50)
    for b in r:
        print(b)
