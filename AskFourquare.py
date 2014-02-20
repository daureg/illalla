#! /usr/bin/python2
# vim: set fileencoding=utf-8
from datetime import datetime
from read_foursquare import Location
import foursquare
from collections import namedtuple
from foursquare_keys import FOURSQUARE_ID as CLIENT_ID
from foursquare_keys import FOURSQUARE_SECRET as CLIENT_SECRET
from persistent import load_var, save_var

Venue = namedtuple('Venue', ['id', 'name', 'loc', 'cats', 'cat', 'stats',
                             'hours', 'price', 'rating', 'createdAt', 'mayor',
                             'tags', 'shortUrl', 'canonicalUrl', 'likes',
                             'likers'])
Categories = namedtuple('Categories', ['id', 'name', 'sub'])


class RequestsMonitor():
    """Request monitor to avoid exceeding API rate limit."""
    window_start = None
    current_load = None
    rate = None

    def __init__(self, rate=5000):
        self.rate = rate

    def more_allowed(self, just_checking=False):
        if self.window_start is None:
            if not just_checking:
                self.window_start = datetime.now()
                self.current_load = 1
            return True
        else:
            if (datetime.now() - self.window_start).total_seconds() > 3600:
                self.window_start = datetime.now()
                self.current_load = 0

        allowed = self.current_load < self.rate
        if not just_checking and allowed:
            self.current_load += 1
        return allowed


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


def venue_profile(client, vid):
    """Return a Venue object from a venue id."""
    # venue = client.venue(vid)['venue']
    venue = load_var('venue')['venue']
    name = venue['name']
    loc = venue['location']
    loc = Location('Point', [loc['lat'], loc['lng']])
    cats = [c['id'] for c in venue['categories']]
    cat = cats.pop(0)
    stats = venue['stats'].values()
    hours = None
    if 'hours' in venue:
        pass
    price = None if 'price' not in venue else venue['price']['tier']
    rating = None if 'rating' not in venue else venue['rating']
    createdAt = datetime.fromtimestamp(venue['createdAt'])
    mayor = int(venue['mayor']['user']['id'])
    tags = venue['tags']
    shortUrl = venue['shortUrl']
    canonicalUrl = venue['canonicalUrl']
    likes = venue['likes']
    likers = None
    if likes['count'] > 0:
        groups = [g['items'] for g in likes['groups']
                  if g['type'] == 'others']
        likers = [int(u['id']) for g in groups for u in g]
        likes = likes['count']
    else:
        likes = 0

    return Venue(vid, name, loc, cats, cat, stats, hours, price, rating,
                 createdAt, mayor, tags, shortUrl, canonicalUrl, likes, likers)


if __name__ == '__main__':
    # import doctest
    # doctest.testmod()
    client = foursquare.Foursquare(CLIENT_ID, CLIENT_SECRET)
    # print(venue_profile(client, ''))
    ft = get_categories()
