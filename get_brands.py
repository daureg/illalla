#! /usr/bin/python2
# vim: set fileencoding=utf-8
import scipy.io as sio
import VenueFeature as vf
import CommonMongo as cm
import persistent as p
DB, CLIENT = cm.connect_to_db('foursquare')
vf.DB = DB
vf.CLIENT = CLIENT
brands = ["mcdonald's", 'starbucks']
import cities as C
starbucks = list(vf.DB.venue.find({'name':
                                   {'$in': ['Starbucks Coffee', 'Starbucks']}},
                                  {'city': 1}))
macdo = list(vf.DB.venue.find({'name': "McDonald's"}, {'city': 1}))
for city in C.SHORT_KEY:
    vindex = set(list(sio.loadmat(city+'_fv')['i']))
    fromdb = set([_['_id'] for _ in macdo if _['city'] == city])
    res = list(fromdb.intersection(vindex))
    p.save_var('{}_{}.my'.format(city, brands[0]), res)
    print('saved {} {} in {}'.format(len(res), brands[0], city))
    fromdb = set([_['_id'] for _ in starbucks if _['city'] == city])
    res = list(fromdb.intersection(vindex))
    p.save_var('{}_{}.my'.format(city, brands[1]), res)
    print('saved {} {} in {}'.format(len(res), brands[1], city))
