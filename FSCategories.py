#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Maintain a tree of Foursquare categories and provide query methods."""
from collections import namedtuple
import persistent as p
import string
Category = namedtuple('Category', ['id', 'name', 'depth', 'sub'])
import bidict
CAT_TO_ID = bidict.bidict({'Venue': '0'})
ID_TO_INDEX = bidict.bidict({'0': 0})


def parse_categories(top_list, depth=0):
    """Recursively build Categories"""
    if len(top_list) == 0:
        return []
    res = []
    for cat in top_list:
        subs = []
        if isinstance(cat, dict) and 'categories' in cat:
            subs = parse_categories(cat['categories'], depth+1)
        id_, name = str(cat['id']), unicode(cat['name'])
        CAT_TO_ID[name] = id_
        res.append(Category(id_, name, depth+1, subs))
    return res


def get_categories(client=None):
    """Return categories list from disk or from Foursquare website using
    client"""
    if client is None:
        raw_cats = p.load_var('raw_categories')['categories']
    else:
        raw_cats = client.venues.categories()
        p.save_var('raw_categories', raw_cats)
        raw_cats = raw_cats['categories']
    cats = Category('0', 'Venue', 0, parse_categories(raw_cats))
    # pylint: disable=E1101
    id_index = [(id_, idx)
                for idx, id_ in enumerate(sorted(CAT_TO_ID.values()))]
    ID_TO_INDEX.update(id_index)
    return cats


def search_categories(cats, query, field=None):
    """Return a category matching query (either by name or id) and its path
    inside cats."""
    if field is None:
        field = choose_type(query)
    if cats[field] == query:
        return cats, [query]
    for sub_category in cats.sub:
        found, path = search_categories(sub_category, query, field)
        if found is not None:
            return found, [cats[field]] + path
    return None, None


def choose_type(query):
    """Return appropriate field index for `query`."""
    if query[0] in string.digits:
        return 0
    return 1

if __name__ == '__main__':
    #pylint: disable=C0103
    ft = get_categories()
    cbar, bpath = search_categories(ft, 'Bar')
