#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Functions used in twitter scrapper main code."""
import functools
from timeit import default_timer as clock


def import_json():
    """Return a json module (first trying ujson then simplejson and finally
    json from standard library)."""
    try:
        import ujson as json
    except ImportError:
        # try:
        #     import simplejson as json
        # except ImportError:
        #     import json
        # I cannot make the others two work with utf-8
        raise
    return json


def log_exception(log, reraise=False):
    """If `func` raises an exception, log it to `log`. By default, assume it's
    not critical and thus resume execution, except if `reraise` is True."""
    def actual_decorator(func):
        """Real decorator, with no argument"""
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            """Wrapper"""
            try:
                return func(*args, **kwds)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                log.exception("")
                if reraise:
                    raise
        return wrapper
    return actual_decorator


class Failures(object):
    """Keep track of Failures."""
    def __init__(self, initial_waiting_time):
        """`initial_waiting_time` is in minutes."""
        self.nb_failures = 0
        self.last_failure = clock()
        self.initial_waiting_time = float(initial_waiting_time)*60.0
        self.waiting_time = self.initial_waiting_time

    def fail(self):
        """Register a new failure and return a reasonable time to wait"""
        if self.has_failed_recently():
            # Hopefully the golden ration will bring us luck next time
            self.waiting_time *= 1.618
        else:
            # reset waiting time
            self.waiting_time = self.initial_waiting_time
        self.nb_failures += 1
        self.last_failure = clock()
        return self.waiting_time

    def has_failed_recently(self, small=3600):
        """Has it failed in the last `small` seconds?"""
        return self.nb_failures > 0 and clock() - self.last_failure < small
