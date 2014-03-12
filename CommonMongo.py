#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Gather some commonly used code related to MongoDB."""
import pymongo
import cities
HOST = 'localhost'
PORT = 27017


def connect_to_db(dbname, host=HOST, port=PORT, client=None):
    """Return a connection to `dbname`, potentially creating and returning a
    client in the process if none is provided."""
    if client is None:
        client = pymongo.MongoClient(host, port)
    return client[dbname], client


def build_query(city=None, venue=True, fields=None, limit=None):
    """Return a template query, that may restrict results to a given `city`, to
    checkin that includes `venue` id or to only `limit` of them. It also
    includes extra attributes defined by `fields`."""
    match = {}
    if isinstance(city, str):
        assert city in cities.SHORT_KEY
        match['city'] = city
    if venue:
        match['lid'] = {'$ne': None}
    project = {}
    fields = fields or []
    for info in fields:
        project[info] = 1
    if not ('id' in fields or '_id' in fields):
        project['_id'] = 0
    query = [{'$match': match}, {'$project': project}]
    if isinstance(limit, int) and limit > 0:
        query.append({'$limit': limit})
    return query
