#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Use collected tweet and AskFourquare to fill user and venue table in the
Database."""
from time import sleep
from datetime import datetime
from threading import Thread
import foursquare
import pymongo
from Queue import Queue
from api_keys import FOURSQUARE_ID as CLIENT_ID
from api_keys import FOURSQUARE_SECRET as CLIENT_SECRET
from RequestsMonitor import RequestsMonitor
from AskFourquare import gather_all_entities_id
from AskFourquare import user_profile, venue_profile
import sys

ENTITY_KIND = 'venue'
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
ENTITIES_QUEUE = Queue(105)
LIMITOR = RequestsMonitor(RATE)
TABLE = []
TO_BE_INSERTED = []
INVALID_ID = []


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
        answers = []
        try:
            answers = list(CLIENT.multi())
        except foursquare.ParamError as e:
            print(str(e))
            invalid = str(e).split('/')[-1].replace(' ', '+')
            answers = individual_query(batch, invalid)

        for a in answers:
            if a is None:
                print('None answer')
            elif not isinstance(a, foursquare.FoursquareException):
                ENTITIES_QUEUE.put(PARSE(a[ENTITY_KIND]))
            else:
                print(a)
                INVALID_ID.append(str(a).split()[1])
        IDS_QUEUE.task_done()


def individual_query(batch, invalid):
    print(batch, invalid)
    answers = []
    for id_ in batch:
        a = None
        if id_ != invalid:
            try:
                a = REQ(id_)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print(sys.exc_info()[1])
        answers.append(a)
    assert len(answers) == len(batch)
    return answers


def entities_putter():
    while True:
        entity = ENTITIES_QUEUE.get()
        if entity is not None:
            TO_BE_INSERTED.append(convert_entity_for_mongo(entity))
        if len(TO_BE_INSERTED) >= 100:
            mongo_insertion()
        ENTITIES_QUEUE.task_done()


def mongo_insertion():
    global TO_BE_INSERTED
    try:
        TABLE.insert(TO_BE_INSERTED, continue_on_error=True)
    except pymongo.errors.DuplicateKeyError:
        pass
    except pymongo.errors.OperationFailure as e:
        print(e.error)
    TO_BE_INSERTED = []

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
    t = Thread(target=entities_getter, name='Query4SQ')
    t.daemon = True
    t.start()
    t = Thread(target=entities_putter, name='InsertDB')
    t.daemon = True
    t.start()
    total_entities = 0
    for batch in gather_all_entities_id(checkins, DB_FIELD, city='chicago',
                                        limit=None):
            IDS_QUEUE.put(batch)
            total_entities += len(batch)

    IDS_QUEUE.join()
    ENTITIES_QUEUE.join()
    mongo_insertion()
    from persistent import save_var
    print('{}/{} invalid id'.format(len(INVALID_ID), total_entities))
    save_var('non_venue_id_{}'.format(hash(datetime.now())), INVALID_ID)
