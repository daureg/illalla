#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Use collected tweet and AskFourquare to fill user and venue table in the
Database."""
from time import sleep
from threading import Thread
import foursquare
import CommonMongo as cm
import Chunker
from Queue import Queue
from api_keys import FOURSQUARE_ID as CLIENT_ID
from api_keys import FOURSQUARE_SECRET as CLIENT_SECRET
from RequestsMonitor import RequestsMonitor
from AskFourquare import gather_all_entities_id
from AskFourquare import user_profile, venue_profile
import sys
import arguments

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
            print(e)
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
        if len(TO_BE_INSERTED) >= 400:
            mongo_insertion()
        ENTITIES_QUEUE.task_done()


def mongo_insertion():
    global TO_BE_INSERTED
    try:
        TABLE.insert(TO_BE_INSERTED, continue_on_error=True)
    except cm.pymongo.errors.DuplicateKeyError:
        pass
    except cm.pymongo.errors.OperationFailure as e:
        print(e, e.code)
    del TO_BE_INSERTED[:]

if __name__ == '__main__':
    REQ = getattr(CLIENT, REQ)
    args = arguments.city_parser().parse_args()
    db = cm.connect_to_db('foursquare', args.host, args.port)[0]
    checkins = db['checkin']
    TABLE = db[ENTITY_KIND]
    if ENTITY_KIND == 'venue':
        TABLE.ensure_index([('loc', cm.pymongo.GEOSPHERE),
                            ('city', cm.pymongo.ASCENDING),
                            ('cat', cm.pymongo.ASCENDING)])
    t = Thread(target=entities_getter, name='Query4SQ')
    t.daemon = True
    t.start()
    t = Thread(target=entities_putter, name='InsertDB')
    t.daemon = True
    t.start()
    total_entities = 0
    city = args.city
    chunker = Chunker.Chunker(foursquare.MAX_MULTI_REQUESTS)
    previous = [e['_id'] for e in TABLE.find({'city': city})]
    latent = gather_all_entities_id(checkins, DB_FIELD, city=city, limit=None)
    print('but already {} {} in DB.'.format(len(previous), ENTITY_KIND))
    new_ones = set(latent).difference(set(previous))
    print('So only {} new ones.'.format(len(new_ones)))
    for batch in chunker(new_ones):
        IDS_QUEUE.put(batch)
        total_entities += len(batch)

    IDS_QUEUE.join()
    ENTITIES_QUEUE.join()
    mongo_insertion()
    from persistent import save_var
    print('{}/{} invalid id'.format(len(INVALID_ID), total_entities))
    print('{}/{} requests'.format(CLIENT.rate_remaining, CLIENT.rate_limit))
    region = city or 'world'
    save_var('non_{}_id_{}'.format(ENTITY_KIND, region), INVALID_ID)
