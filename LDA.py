#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Topic modeling using [gensim](http://radimrehurek.com/gensim/)"""
import CommonMongo as cm
from datetime import datetime as dt
import logging
import persistent as p
from gensim import corpora, models


def get_tags(city, begin, client):
    """Iterator over tags from photos in `city` since `begin`."""
    return client.world.photos.find({'hint': city, 'taken': {'$gte': begin}},
                                    {'tags': 1, '_id': 0})


def build_dico(city, begin, client):
    technical = ['blackandwhite', 'longexposure', 'lomofi', 'mobile', 'bw',
                 'cameraphone', 'lofi', 'lowlight', 'geotagged', 'xproii',
                 'nikon', 'd200', 'square', 'instagramapp', 'squareformat',
                 'iphoneography', 'iphone', 'colorvibefilter', 'tonemapped',
                 'chameleonfilter', 'hdr' 'noflash', 'photo', 'iphone4',
                 'iphone5', 'fujix100', 'd700', 'f22', 'photoshop',
                 'photography', 'pictures', 'f18', 'canonef24105mmf4lisusm',
                 'nikond90', 'nikond700']
    dictionary = corpora.Dictionary(p['tags']
                                    for p in get_tags(city, client, begin))
    dictionary.filter_extremes(no_below=30, no_above=0.4, keep_n=None)
    dictionary.compactify()
    stop_ids = [dictionary.token2id[stopword] for stopword in technical
                if stopword in dictionary.token2id]
    good_words = [_[0] for _ in p.load_var(city+'_tag_support')]
    good_ids = [dictionary.token2id[goodword] for goodword in good_words
                if goodword in dictionary.token2id]
    dictionary.filter_tokens(bad_ids=stop_ids, good_ids=good_ids)
    # remove gaps in id sequence after words that were removed
    dictionary.compactify()
    print(dictionary)
    dictionary.save(city+'_flickr.dict')
    return dictionary


class PhotoCorpus(object):
    """Corpus of tags as bag-of-word"""
    def __init__(self, city, client, dictionary, begin):
        self.query = {'hint': city, 'taken': {'$gte': begin}}
        self.db = client.world.photos
        self.dictionary = dictionary

    def __len__(self):
        return self.db.find(self.query, {'_id': 1}).count()

    def __iter__(self):
        for photo in self.db.find(self.query, {'tags': 1, '_id': 0}):
            yield self.dictionary.doc2bow(photo['tags'])

    def save(self):
        corpora.MmCorpus.serialize(city+'_corpus.mm', self)


def find_topics(city, client, num_topics, begin):
    dico = build_dico(city, client, begin)
    bow_photos = PhotoCorpus(city, client, dico, begin)
    bow_photos.save()

    lda = models.LdaModel(bow_photos, id2word=dico, num_topics=num_topics,
                          passes=5, iterations=500, eval_every=5)
    lda_photos = lda[bow_photos]
    print(lda.show_topics(10))
    lda.save(city+'.lda')
    return lda

if __name__ == '__main__':
    # pylint: disable=C0103
    import arguments
    args = arguments.city_parser().parse_args()
    city = args.city
    logging.basicConfig(filename=city+'_lda.log',
                        format='%(asctime)s [%(levelname)s]: %(message)s',
                        level=logging.INFO)
    db, client = cm.connect_to_db('foursquare', args.host, args.port)
    lda = find_topics(city, client, num_topics=80, begin=dt(2008, 1, 1))
