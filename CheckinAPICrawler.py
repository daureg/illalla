#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Get checkin info by expanding 4sq.com short url with bit.ly and then
requesting Foursquare API."""

import itertools
import ssl
import bitly_api
import foursquare
import Chunker
import logging
logging.basicConfig(filename='tweets.log', level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s')
import twitter_helper as th
CHECKIN_URL = th.CHECKIN_URL
from api_keys import BITLY_TOKEN
from api_keys import FOURSQUARE_ID2 as CLIENT_ID
from api_keys import FOURSQUARE_SECRET2 as CLIENT_SECRET
BITLY_SIZE = 15


def get_id_and_signature(url):
    """Potentially extract checkin id and signature from `url`."""
    components = None if not url else CHECKIN_URL.search(url)
    if not components:
        return (None, None)
    return components.groups()


class FoursquareDown(Exception):
    """Signal that even after waiting a long time, request to Foursquare still
    result in ServerError."""
    pass


class CheckinAPICrawler(object):
    """Get checkins info."""
    def __init__(self):
        self.bitly = bitly_api.Connection(access_token=BITLY_TOKEN)
        self.client = foursquare.Foursquare(CLIENT_ID, CLIENT_SECRET)
        self.bitly_batch = Chunker.Chunker(BITLY_SIZE)
        self.failures = th.Failures(initial_waiting_time=1.8)

    def checkins_from_url(self, urls):
        """Return info from all url in `urls`"""
        res = []
        for batch in self.bitly_batch(urls):
            try:
                res.extend(self.get_checkins_info(batch))
            except FoursquareDown:
                logging.exception("Foursquare not responding")
                return None
        return res

    def expand_urls(self, urls):
        """Use Bitly to expand short link in `urls`.
        Return a list of (checkin id, signature)."""
        try:
            expanded = [res.get('long_url', None)
                        for res in self.bitly.expand(link=urls)]
        except bitly_api.BitlyError:
            logging.exception("Error expanding URL")
            # Could also wait here, but actually, I have never seen it happen.
            # So let's trust bitly reliability for now
            expanded = itertools.repeat(None, BITLY_SIZE)
        return [get_id_and_signature(url) for url in expanded]

    # TODO: Swarm checkin don't have signature in their public URL and they
    # don't need it because there is a new API call that deals with that:
    # https://developer.foursquare.com/docs/checkins/resolve
    # Check if it's already in foursquare python api
    def query_foursquare(self, id_and_sig):
        """Request Foursquare to get raw info about checkins in `id_and_sig`"""
        for cid, sig in id_and_sig:
            if cid:
                self.client.checkins(cid, {'signature': sig}, multi=True)
        try:
            return self.client.multi()
        except (foursquare.FoursquareException, ssl.SSLError):
            logging.exception("Error requesting batch checkins")
            waiting_time = self.failures.fail()
            if self.failures.recent_failures >= 5:
                raise FoursquareDown
            msg = 'Will wait for {:.0f} seconds'.format(waiting_time)
            logging.info(msg)
            self.failures.do_sleep()

    def get_checkins_info(self, urls):
        """Return info from a batch of `urls`"""
        id_and_sig = self.expand_urls(urls)
        raw_checkins = self.query_foursquare(id_and_sig)

        res = []
        for cid, sig in id_and_sig:
            if not cid:
                res.append(None)
                continue
            try:
                raw_checkin = raw_checkins.next()
            except foursquare.ServerError as oops:
                logging.exception('error in getting next checkin')
                if 'status' in str(oops):
                    waiting_time = self.failures.fail()
                    if self.failures.recent_failures >= 5:
                        raise FoursquareDown
                    msg = 'Will wait for {:.0f} seconds'.format(waiting_time)
                    logging.info(msg)
            if isinstance(raw_checkin, foursquare.FoursquareException):
                msg = 'Weird id: {}?s={}\n{}'.format(cid, sig,
                                                     str(raw_checkin))
                logging.warn(msg)
                res.append(None)
            else:
                parsed = th.parse_json_checkin(raw_checkin)
                checkin_info = None
                if parsed:
                    checkin_info = (cid + '?s=' + sig, ) + parsed
                res.append(checkin_info)
        return res

if __name__ == '__main__':
    # pylint: disable=C0103
    crawler = CheckinAPICrawler()
    test_urls = ['http://4sq.com/1n6608z', u'http://4sq.com/1h1Ii5S',
                 'http://4sq.com/FAKE666', 'http://4sq.com/h2UIDl',
                 'http://4sq.com/gNlqGb', 'http://4sq.com/eISsW0',
                 'http://4sq.com/fLc5gJ', 'http://4sq.com/eCaKGm']
    for ck in crawler.checkins_from_url(test_urls):
        print(ck)
