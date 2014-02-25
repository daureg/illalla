#! /usr/bin/python2
# vim: set fileencoding=utf-8
import persistent as p


class CheckinCorrector():

    def __init__(self):
        # query lid to get tweet id and build a map
        # go through text file and retrieve msg  (many/h)
        # expand again url but return directly last split component
        # extract id and sig (> 10 000 / h)
        # make multi call to get correct lid (2500 / h)
        # update the checkin in DB
        # save these new lid because we need to build their profile
        pass

if __name__ == '__main__':
    checkin_ids = [u for u in p.load_var('non_venue_id')
                   if len(u) == 24]
