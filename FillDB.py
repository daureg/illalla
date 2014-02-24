#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Use collected tweet and AskFourquare to fill user and venue table in the
Database."""
from time import sleep
from threading import Thread
import foursquare
import pymongo
from Queue import Queue
from api_keys import FOURSQUARE_ID as CLIENT_ID
from api_keys import FOURSQUARE_SECRET as CLIENT_SECRET
from AskFourquare import gather_all_entities_id, RequestsMonitor
from AskFourquare import user_profile, venue_profile

ENTITY_KIND = 'user'
if ENTITY_KIND == 'venue':
    RATE = 5000
    REQ = 'venues'
    DB_FIELD = 'lid'
    PARSE = venue_profile
elif ENTITY_KIND == 'user':
    RATE = 500
    REQ = 'users'
    DB_FIELD = 'uid'
    PARSE = user_profile
else:
    raise ValueError(ENTITY_KIND + ' is unknown')
CLIENT = foursquare.Foursquare(CLIENT_ID, CLIENT_SECRET)
IDS_QUEUE = Queue(5)
ENTITIES_QUEUE = Queue(100)
LIMITOR = RequestsMonitor(RATE)
TABLE = []
TO_BE_INSERTED = []


def convert_entity_for_mongo(entity):
    suitable = entity._asdict()
    suitable['_id'] = suitable['id']
    del suitable['id']
    return suitable


def entities_getter():
    while True:
        batch = IDS_QUEUE.get()
        go, wait = LIMITOR.more_allowed(CLIENT)
        if not go:
            sleep(wait + 3)
        for id_ in batch:
            REQ(id_, multi=True)
        for a in CLIENT.multi():
            if not isinstance(a, foursquare.FoursquareException):
                ENTITIES_QUEUE.put(PARSE(a[ENTITY_KIND]))
            else:
                print(a)
        IDS_QUEUE.task_done()


def entities_putter():
    while True:
        entity = ENTITIES_QUEUE.get()
        TO_BE_INSERTED.append(convert_entity_for_mongo(entity))
        if len(TO_BE_INSERTED) >= 20:
            mongo_insertion()
        ENTITIES_QUEUE.task_done()


def mongo_insertion():
    return
    global TO_BE_INSERTED
    try:
        TABLE.insert(TO_BE_INSERTED, continue_on_error=True)
        TO_BE_INSERTED = []
    except pymongo.errors.OperationsFailure as e:
        print(e.error)

if __name__ == '__main__':
    REQ = getattr(CLIENT, REQ)
    mongo_client = pymongo.MongoClient('localhost', 27017)
    db = mongo_client['foursquare']
    checkins = db['checkin']
    TABLE = db[ENTITY_KIND]
    if ENTITY_KIND == 'venue':
        TABLE.ensure_index([('loc', pymongo.GEOSPHERE),
                            ('_id', pymongo.ASCENDING),
                            ('city', pymongo.ASCENDING),
                            ('tags', pymongo.ASCENDING),
                            ('cat', pymongo.ASCENDING)])
    elif ENTITY_KIND == 'user':
        TABLE.ensure_index([('_id', pymongo.ASCENDING)])
    t = Thread(target=entities_getter)
    t.daemon = True
    t.start()
    t = Thread(target=entities_putter)
    t.daemon = True
    t.start()
    for batch in gather_all_entities_id(checkins, DB_FIELD, city='helsinki',
                                        limit=50):
        IDS_QUEUE.put(batch)

    IDS_QUEUE.join()
    ENTITIES_QUEUE.join()
    mongo_insertion()
