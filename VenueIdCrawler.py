#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Send request to get venue id from short url checkins"""
from timeit import default_timer as clock
import pycurl
import cStringIO as cs
import re
import logging
import os
import urlparse
logging.basicConfig(filename=os.path.expanduser('~/venue.log'),
                    level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s')
POOL_SIZE = 30
import twitter_helper as th
CHECKIN_URL = th.CHECKIN_URL


class VenueIdCrawler(object):
    cpool = None
    bpool = None
    claim_id = None
    multi = None
    pool_size = 0
    connections = 0
    results = {None: None}
    errors = []
    use_network = False
    checkin_url = None

    def __init__(self, pre_computed=None, use_network=True,
                 pool_size=POOL_SIZE):
        assert isinstance(pool_size, int) and pool_size > 0
        self.pool_size = pool_size
        self.claim_id = re.compile(r'claim\?vid=([0-9a-f]{24})')
        self.checkin_url = CHECKIN_URL
        self.fs_id = re.compile(r'[0-9a-f]{24}')
        venue_url = r'venueName"><a href="/v/[^/]+/([0-9a-f]{24})'
        self.venue_url = re.compile(venue_url)
        self.multi = pycurl.CurlMulti()
        self.cpool = [pycurl.Curl() for _ in range(self.pool_size)]
        self.bpool = [cs.StringIO() for _ in range(self.pool_size)]
        self.use_network = use_network
        for i, c in enumerate(self.cpool):
            c.buf = self.bpool[i]
            c.setopt(pycurl.FOLLOWLOCATION, 1)
            c.setopt(pycurl.MAXREDIRS, 6)
            c.setopt(pycurl.WRITEFUNCTION, c.buf.write)
            c.setopt(pycurl.CONNECTTIMEOUT, 8)
            c.setopt(pycurl.TIMEOUT, 14)
        if pre_computed is not None:
            self.results = pre_computed
            self.results['None'] = None

    def venue_id_from_urls(self, urls):
        start = clock()
        nb_urls = len(urls)
        batch = []
        if self.use_network:
            for i, u in enumerate(urls):
                if u is not None and u not in self.results:
                    batch.append(u)
                if len(batch) == self.pool_size or i == nb_urls - 1:
                    self.prepare_request(batch)
                    self.perform_request()
                    del batch[:]
            report = 'query {} urls in {:.2f}s, {} (total) errors'
            logging.info(report.format(len(urls), clock() - start,
                                       len(self.errors)))
        return [None if u not in self.results else self.results[u]
                for u in urls]

    def prepare_request(self, urls):
        assert len(urls) <= self.pool_size
        for i, u in enumerate(urls):
            u = str(u)
            self.cpool[i].url = u
            self.cpool[i].buf.truncate(0)
            try:
                self.cpool[i].setopt(pycurl.URL, u)
            except TypeError:
                logging.error('curl fail at ' + str(u))
                continue
            self.multi.add_handle(self.cpool[i])
            self.connections += 1

    def perform_request(self):
        while self.connections > 0:
            status = self.multi.select(0.3)
            if status == -1:  # timeout
                continue
            if status > 0:
                self.empty_queue()
            performing = True
            while performing:
                status, self.connections = self.multi.perform()
                if status is not pycurl.E_CALL_MULTI_PERFORM:
                    performing = False
        self.empty_queue()

    def empty_queue(self):
        _, ok, ko = self.multi.info_read()
        for failed in ko:
            self.errors.append((failed[0].url, failed[1]))
            self.multi.remove_handle(failed[0])
        for success in ok:
            self.results[success.url] = self.get_venue_id(success)
            self.multi.remove_handle(success)

    def get_venue_id(self, curl_object):
        if curl_object.getinfo(pycurl.HTTP_CODE) != 200:
            return None
        url = curl_object.getinfo(pycurl.EFFECTIVE_URL)
        domain = urlparse.urlparse(url).netloc
        if 'foursquare' not in domain:
            return None
        id_ = url.split('/')[-1]
        if len(id_) == 24 and self.fs_id.match(id_):
            return id_
        if len(id_) == 54 and self.checkin_url.match(id_):
            match = self.venue_url.search(curl_object.buf.getvalue())
            return None if not match else match.group(1)
        # we probably got a vanity url like https://foursquare.com/radiuspizza
        # thus we look at its content and try to find the link to claim this
        # venue, because it contains the numerical id.
        match = self.claim_id.search(curl_object.buf.getvalue())
        return None if not match else match.group(1)


def venue_id_from_url(c, url):
    """
    Return the id of the venue associated with short url
    >>> venue_id_from_url(pycurl.Curl(), 'http://4sq.com/31ZCjK')
    '44d17cecf964a5202b361fe3'
    """
    c.reset()
    c.setopt(pycurl.URL, url)
    c.setopt(pycurl.FOLLOWLOCATION, 1)
    c.setopt(pycurl.MAXREDIRS, 6)
    c.setopt(pycurl.NOBODY, 1)
    c.perform()
    if c.getinfo(pycurl.HTTP_CODE) == 200:
        return c.getinfo(pycurl.EFFECTIVE_URL).split('/')[-1]
    return None

TEST_DATA = {
    # # non 200 code
    'http://bit.ly/9d0NTH': None,
    'http://bit.ly/bFMzkX': None,
    'http://bit.ly/bqKggn': None,
    'http://bit.ly/bCiLRc': None,
    # non 4SQ page
    'http://nie.mn/fhgQ79': None,
    'http://nyti.ms/aXIS2O': None,
    'http://nyti.ms/coIHaC': None,
    # check in page
    'http://4sq.com/h2UIDl': '45ca42e1f964a5207a421fe3',
    'http://4sq.com/gNlqGb': '46cf28b9f964a520454a1fe3',
    'http://4sq.com/eISsW0': '4c73269df3279c74f15eb12d',
    'http://4sq.com/fLc5gJ': '3fd66200f964a52072e61ee3',
    'http://4sq.com/eCaKGm': '4d1759fd25cda143c24876d6',
    'http://4sq.com/1fUl5jB': '4b3b5b13f964a520d97225e3',
    'http://4sq.com/1n6608z': '4e5b759eb0fbdca30a171b16',
    u'http://4sq.com/1h1Ii5S': '4bbcab08a0a0c9b654971a0f',
    u'http://4sq.com/1qjygUa': '49bb36fcf964a520dc531fe3',
    # venue page
    'http://4sq.com/ccIVP3': '4e57dbd3227131507c9381df',
    'http://4sq.com/9nNdot': '4a270788f964a520268e1fe3',
    'http://4sq.com/b6jdBy': '4b819c57f964a5208ab230e3',
    'http://4sq.com/7z3PQ9': '4af2c6f1f964a52075e821e3',
    'http://4sq.com/d8fVOz': '4b50d2b3f964a520bd3327e3',
    # vanity url
    'http://4sq.com/8dwV1O': '4a8212dcf964a5207cf81fe3',
    # non venue 4SQ page
    'http://goo.gl/WLoshz': None
}

if __name__ == '__main__':
    r = VenueIdCrawler()
    query_url = TEST_DATA.keys()
    res = r.venue_id_from_urls(query_url)
    res_dict = {u: i for u, i in zip(query_url, res)}
    print('correct: {}/{}'.format(len(res), len(TEST_DATA)))
    for g, m in zip(sorted(TEST_DATA.items()), sorted(res_dict.items())):
        print(g, m)
