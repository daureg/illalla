#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Functions used in twitter scrapper main code."""
import functools


def import_json():
    """Return a json module (first trying ujson then simplejson and finally
    json from standard library)."""
    try:
        import ujson as json
    except ImportError:
        try:
            import simplejson as json
        except ImportError:
            import json
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
