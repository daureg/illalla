#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Get checkin info by expanding 4sq.com short url with bit.ly and then
requesting Foursquare API."""

import itertools
import ssl
import bitly_api
import foursquare
import Chunker
import CheckinCrawler as cc
from VenueIdCrawler import CHECKIN_URL
from api_keys import BITLY_TOKEN
from api_keys import FOURSQUARE_ID as CLIENT_ID
from api_keys import FOURSQUARE_SECRET as CLIENT_SECRET
BITLY_SIZE = 15


def get_id_and_signature(url):
    """Potentially extract checkin id and signature from `url`."""
    components = None if not url else CHECKIN_URL.search(url)
    if not components:
        return (None, None)
    return components.groups()


class CheckinAPICrawler(object):
    """Get checkins info."""
    def __init__(self):
        self.bitly = bitly_api.Connection(access_token=BITLY_TOKEN)
        self.client = foursquare.Foursquare(CLIENT_ID, CLIENT_SECRET)
        self.bitly_batch = Chunker.Chunker(BITLY_SIZE)

    def checkins_from_url(self, urls):
        """Return info from all url in `urls`"""
        res = []
        for batch in self.bitly_batch(urls):
            res.extend(self.get_checkins_info(batch))
        return res

    def get_checkins_info(self, urls):
        """Return info from a batch of `urls`"""
        try:
            expanded = [res.get('long_url', None)
                        for res in self.bitly.expand(link=urls)]
        except bitly_api.BitlyError as oops:
            expanded = itertools.repeat(None, BITLY_SIZE)
        id_and_sig = [get_id_and_signature(url) for url in expanded]
        res = []
        for cid, sig in id_and_sig:
            if cid:
                self.client.checkins(cid, {'signature': sig}, multi=True)
        try:
            raw_checkins = self.client.multi()
        except (foursquare.ServerError, ssl.SSLError) as oops:
            print(oops)
            # TODO: do something more clever
            raw_checkins = itertools.repeat(None, BITLY_SIZE)

        for cid, sig in id_and_sig:
            if not cid:
                res.append(None)
            else:
                parsed = cc.parse_json_checkin(raw_checkins.next())
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
