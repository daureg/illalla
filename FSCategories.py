#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Maintain a tree of Foursquare categories and provide query methods."""
from collections import namedtuple
import persistent as p
import string
Categories = namedtuple('Categories', ['id', 'name', 'sub'])


def parse_categories(top_list):
    if len(top_list) == 0:
        return []
    res = []
    for cat in top_list:
        subs = []
        if isinstance(cat, dict) and 'categories' in cat:
            subs = parse_categories(cat['categories'])
        res.append(Categories(cat['id'], cat['shortName'], subs))
    return res


def get_categories(client=None):
    """Return categories list from disk or from Foursquare website using
    client"""
    if client is None:
        raw_cats = p.load_var('raw_categories')['categories']
    else:
        raw_cats = client.categories()['categories']
    cats = Categories('0', '_', parse_categories(raw_cats))
    p.save_var('categories', cats)
    return cats


def search_categories(cats, query, field=None):
    """Return a category matching query (either name or id) and its path inside
    cats."""
    if field is None:
        field = 0 if query[0] in string.digits else 1
    if cats[field] == query:
        return cats, [query]
    for sub_category in cats.sub:
        found, path = search_categories(sub_category, query, field)
        if found is not None:
            return found, [cats[field]] + path
    return None, None

if __name__ == '__main__':
    pass
