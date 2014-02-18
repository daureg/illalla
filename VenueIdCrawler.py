#! /usr/bin/python2
# vim: set fileencoding=utf-8
from timeit import default_timer as clock
from itertools import izip_longest
import pycurl
POOL_SIZE = 30


def grouper(iterable, n, fillvalue=None):
    """
    from http://docs.python.org/2/library/itertools.html#recipes
    Collect data into fixed-length chunks or blocks
    >>> list(grouper('ABCDEFG', 3, 'x'))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')]
    """
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


class VenueIdCrawler():

    cpool = None
    multi = None
    pool_size = 0
    remaining = 0
    results = {}
    errors = []

    def __init__(self, pool_size=POOL_SIZE):
        assert isinstance(pool_size, int) and pool_size > 0
        self.pool_size = pool_size
        self.multi = pycurl.CurlMulti()
        self.cpool = [pycurl.Curl() for _ in range(self.pool_size)]
        for c in self.cpool:
            c.setopt(pycurl.FOLLOWLOCATION, 1)
            c.setopt(pycurl.MAXREDIRS, 6)
            c.setopt(pycurl.NOBODY, 1)

    def venue_id_from_url(self, urls):
        start = clock()
        for batch in grouper(urls, self.pool_size):
            self.prepare_request(batch)
            self.perform_request()
        report = 'query {} urls in {:.2f}s, {} and {} errors'
        fails = [1 for v in self.results.values() if v is None]
        print(report.format(len(urls), clock() - start, len(self.errors),
                            len(fails)))
        self.remaining = 0
        return self.results

    def prepare_request(self, urls):
        assert len(urls) == self.pool_size
        for i, u in enumerate(urls):
            if u is not None:
                self.cpool[i].setopt(pycurl.URL, u)
                self.cpool[i].url = u
                self.multi.add_handle(self.cpool[i])
                self.remaining += 1

    def perform_request(self):
        while self.remaining > 0:
            status = self.multi.select(0.3)
            if status == -1:  # timeout
                continue
            if status > 0:
                self.empty_queue()
            performing = True
            while performing:
                status, self.remaining = self.multi.perform()
                if status is not pycurl.E_CALL_MULTI_PERFORM:
                    performing = False
        self.empty_queue()

    def empty_queue(self):
        _, ok, ko = self.multi.info_read()
        for failed in ko:
            self.errors.append(failed[1])
            self.multi.remove_handle(failed[0])
        for success in ok:
            self.results[success.url] = VenueIdCrawler.get_venue_id(success)
            self.multi.remove_handle(success)

    @staticmethod
    def get_venue_id(curl_object):
        if curl_object.getinfo(pycurl.HTTP_CODE) == 200:
            return curl_object.getinfo(pycurl.EFFECTIVE_URL).split('/')[-1]
        return None


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

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    from persistent import load_var
    urls = ['http://4sq.com/1ZNmiJ', 'http://4sq.com/2z5p82',
            'http://4sq.com/31ZCjK', 'http://4sq.com/3iFEGH',
            'http://4sq.com/4ADi6k', 'http://4sq.com/4FFBOp',
            'http://4sq.com/4k1L7c', 'http://4sq.com/5DPD7A',
            'http://4sq.com/5NbLrk', 'http://4sq.com/67XL9k',
            'http://4sq.com/75SfNv', 'http://4sq.com/7DAVph',
            'http://4sq.com/7JaOFa', 'http://4sq.com/7sG6jW',
            'http://4sq.com/7yoatn', 'http://4sq.com/81eMd8',
            'http://4sq.com/8bg7q4', 'http://4sq.com/8gFJww',
            'http://4sq.com/8KuiUi', 'http://4sq.com/9cg9xg',
            'http://4sq.com/9E7CnF', 'http://4sq.com/9GoW57',
            'http://4sq.com/9hM2SE', 'http://4sq.com/9qN20H',
            'http://4sq.com/9yHXf0', 'http://4sq.com/alRmoX',
            'http://4sq.com/c7m20Y', 'http://4sq.com/cDCxsE',
            'http://4sq.com/ck4qtA', 'http://4sq.com/cXUN8F',
            'http://4sq.com/cYesTR', 'http://4sq.com/dlIcOc',
            'http://4sq.com/dy5um7', 'http://4sq.com/dz96EL',
            'http://t.co/3kGBr2l', 'http://t.co/90Fh8ks',
            'http://t.co/9COjxhy', 'http://t.co/axfykVI',
            'http://t.co/djzPO2S', 'http://t.co/gFaEd0N',
            'http://t.co/HEAq94l', 'http://t.co/Hl8jVOi',
            'http://t.co/iQ6yeYi', 'http://t.co/MkpAVDb',
            'http://t.co/Mu61K9b', 'http://t.co/n4kqQR0',
            'http://t.co/NN1xkiq', 'http://t.co/T88WGpO',
            'http://t.co/WQn3bFf', 'http://t.co/y8qxjsT',
            'http://t.co/ycbb5kt']
    gold = load_var('gold_url')
    start = clock()
    r = VenueIdCrawler()
    res = r.venue_id_from_url(urls[:len(gold)])
    print('{:.2f}s'.format(clock() - start))
    shared_items = set(gold.items()) & set(res.items())
    print('match with gold: {}/{}'.format(len(shared_items), len(gold)))
