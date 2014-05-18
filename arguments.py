#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Provide a standard argument parser."""
import argparse


def get_parser(desc=None):
    """build a parser with host and port number"""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-p", "--port", help="MongoDB port number",
                        type=lambda x: valid_number(x, 1023, 64536),
                        default=27017)
    parser.add_argument("--host", help="MongoDB host name",
                        default="localhost")
    return parser


def valid_city(city):
    """Ensure city argument is valid"""
    import cities
    city = cities.short_name(city)
    if city == 'whole' or city in cities.SHORT_KEY:
        return city
    raise argparse.ArgumentTypeError('{} is not a known city'.format(city))


def city_parser(desc=None):
    """default parser plus city name"""
    parser = get_parser(desc)
    parser.add_argument("city", help="city name", type=valid_city)
    return parser


def two_cities(desc=None):
    """default parser plus 2 city names"""
    parser = get_parser(desc)
    parser.add_argument("origin", help="The city where you choose venues",
                        type=valid_city)
    parser.add_argument("dest", help="The city where to find venues",
                        type=valid_city)
    return parser


def valid_number(number, lower, upper, ntype=int):
    """Ensure numeric argument is within bounds"""
    try:
        number = ntype.__call__(number)
    except ValueError:
        raise argparse.ArgumentTypeError('{} is not a number'.format(number))
    if lower < number < upper:
        return number
    msg = '{} is not between {} and {}'.format(number, lower+1, upper-1)
    raise argparse.ArgumentTypeError(msg)


def tweets_parser():
    """default parser plus run time duration"""
    parser = get_parser('stream Twitter to find checkins')
    parser.add_argument("duration", help="how many hours to run",
                        type=lambda x: valid_number(x, 0, 24, float))
    parser.add_argument("-m", "--mongodb", action="store_true", default=False,
                        help="Store checkins in a Mongo database")
    return parser

if __name__ == '__main__':
    ARGS = tweets_parser()
    ARGS = ARGS.parse_args()
    if ARGS.mongodb:
        print('mongo')
    else:
        print('text')
