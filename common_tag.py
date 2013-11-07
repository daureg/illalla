#! /usr/bin/python2
# vim: set fileencoding=utf-8
import more_query as mq

if __name__ == '__main__':
    sf = mq.get_top_tags(300)
    us = mq.get_top_tags(300, 'us_tags.dat')
    ca = mq.get_top_tags(300, 'ca_tags.dat')
    sfca = [s for s in sf if s in ca]
    sfus = [s for s in sf if s in us]
    sfusca = [s for s in sfca if s in us]
