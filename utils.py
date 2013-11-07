#! /usr/bin/python2
# vim: set fileencoding=utf-8


def to_css_hex(color):
    r = '#'
    for i in color[:-1]:
        c = hex(int(255*i))[2:]
        if len(c) == 2:
            r += c
        else:
            r += '0' + c
    return r
