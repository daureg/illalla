#! /usr/bin/python2
# vim: set fileencoding=utf-8
from api_keys import FOURSQUARE_ID as CLIENT_ID
from api_keys import FOURSQUARE_SECRET as CLIENT_SECRET
import foursquare
import read_foursquare as rf
import RequestsMonitor as rm
import VenueIdCrawler as vc
import cities
import Chunker
import persistent as p
import pymongo
import sys
import time
import Queue
import threading
from itertools import izip
from bisect import bisect_left


class CheckinCorrector(object):
    tweets_id = []
    cids = None
    checkinDB = None
    city = None
    checkins_short_url = None
    crawler = None
    queue = None
    new_venues = []
    updator = None
    infile = None

    def __init__(self, cids, checkinDB, infile, city=None):
        import os
        assert os.path.exists(infile)
        self.cids = cids
        self.checkinDB = checkinDB
        self.city = city
        self.crawler = vc.VenueIdCrawler(use_network=True, full_url=True)
        self.queue = Queue.Queue(40*foursquare.MAX_MULTI_REQUESTS)
        self.updator = threading.Thread(target=self.update_checkin_DB,
                                        name="DB-update")
        self.updator.daemon = True
        self.infile = infile

    def correct(self):
        self.get_tweets()
        self.get_msg_from_url()
        self.expand_url()
        self.updator.start()
        self.url_to_venue_id()
        self.queue.join()
        self.save_new_lid()

    def get_tweets(self):
        """query lid to get faulty tweet id"""
        chunker = Chunker.Chunker(500)
        for batch in chunker(self.cids):
            query = [
                {'$match': {'lid': {'$in': batch}}},
                {'$project': {'lid': 1}},
            ]
            if isinstance(self.city, str) and self.city in cities.SHORT_KEY:
                query[0]['$match']['city'] = self.city
            res = self.checkinDB.aggregate(query)['result']
            self.tweets_id = sorted([c['_id'] for c in res])

    def is_relevant(self, tweet_id):
        i = bisect_left(self.tweets_id, tweet_id)
        return i != len(self.tweets_id) and self.tweets_id[i] == tweet_id

    def get_msg_from_url(self):
        """ go through text file and retrieve msg (many per hour)"""
        self.checkins_short_url = {}
        with open(self.infile) as f:
            for line in f:
                data = line.strip().split('\t')
                if len(data) is not 7:
                    continue
                tid, msg = int(data[1]), data[5]
                if self.is_relevant(tid):
                    url = rf.extract_url_from_msg(msg)
                    self.checkins_short_url[tid] = url

    def expand_url(self):
        """expand again url but return directly last split component
        extract id and sig (> 10 000 per hour)"""
        chunker = Chunker.Chunker(33*vc.POOL_SIZE)
        for tweets in chunker(self.checkins_short_url.iteritems()):
            short_urls = [t[1] for t in tweets]
            long_urls = self.crawler.venue_id_from_urls(short_urls)
            for long_url, tweet in izip(long_urls, tweets):
                self.checkins_short_url[tweet[0]] = long_url

    def url_to_venue_id(self):
        """make multi call to get correct lid (2500 per hour)"""
        limitor = rm.RequestsMonitor(500)
        client = foursquare.Foursquare(CLIENT_ID, CLIENT_SECRET)
        processed = 0
        tids = []
        for tweet, checkin in self.checkins_short_url.iteritems():
            c_id, sig = checkin
            client.checkins(c_id, {'signature': sig}, multi=True)
            tids.append(tweet)
            processed += 1
            if processed % foursquare.MAX_MULTI_REQUESTS == 0:
                self.make_foursquare_requests(tids, client, limitor)

        self.make_foursquare_requests(tids, client, limitor)

    def make_foursquare_requests(self, tids, client, limitor):
        """actually perform Foursquare call (and wait if necessary)"""
        if len(tids) == 0:
            return
        failed = lambda x: isinstance(x, foursquare.FoursquareException) or \
            'checkin' not in x or 'venue' not in x['checkin']
        go, waiting = limitor.more_allowed(client)
        if not go:
            time.sleep(waiting + 3)
        print('do batch')
        try:
            answers = [r['checkin']['venue']['id']
                       for r in client.multi() if not failed(r)]
            for tid, lid in zip(tids, answers):
                self.queue.put((tid, lid))
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print(sys.exc_info()[1])
        finally:
            del tids[:]

    def update_checkin_DB(self):
        """update the checkin in DB"""
        while True:
            tid, lid = self.queue.get()
            self.new_venues.append(lid)
            try:
                self.checkinDB.update({'_id': tid}, {'$set': {'lid': lid}})
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print(sys.exc_info()[1])
            self.queue.task_done()

    def save_new_lid(self):
        """save these new lid because we need to build their profile"""
        region = 'world' if self.city is None else self.city
        id_ = str(hash(self.cids[0]))[:5]
        output = 'new_venue_id_{}_{}'.format(id_, region)
        p.save_var(output, set(self.new_venues))

if __name__ == '__main__':
    city = 'chicago'
    checkin_ids = [u for u in p.load_var('non_venue_id_'+city)
                   if len(u) == 24 and u.startswith('4')]
    mongo_client = pymongo.MongoClient('localhost', 27017)
    db = mongo_client['foursquare']
    cc = CheckinCorrector(checkin_ids, db['checkin'], 'xaa', city)
    cc.correct()
