#! /usr/bin/python2
# vim: set fileencoding=utf-8
from collections import OrderedDict
import codecs


def read_top_tags(n=200, filename='sftags.dat'):
    with codecs.open(filename, 'r', 'utf8') as f:
        lines = [i.strip().split() for i in f.readlines()[1:n+1]]
        # see http://stackoverflow.com/a/1747827 for python 2.7+
        tags = OrderedDict((tag, int(count)) for tag, count in lines)
    return tags.keys()

with codecs.open('dimred.m', 'w', 'utf8') as f:
    f.write("""
X = zeros(200, 40000);
X1 = zeros(200, 10000);
X2 = zeros(200, 2500);""")
    for i, t in enumerate(read_top_tags(200, '../nsf_tag.dat')):
        f.write(u"""
load('freq_200_{}.mat');
X({},:)=c';
X1({},:)=freqred(c');
X2({},:)=freqred(X1({},:));""".format(t, i+1, i+1, i+1, i+1))

    for x in ['X', 'X1', 'X2']:
        f.write("\nsave('-v7', '{}.mat', '{}');".format(x, x))
