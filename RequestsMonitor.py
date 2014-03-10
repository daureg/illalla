#! /usr/bin/python2
# vim: set fileencoding=utf-8
from datetime import datetime, timedelta
ONE_HOUR = timedelta(hours=1)


class RequestsMonitor():
    """Request monitor to avoid exceeding API rate limit."""
    window_start = None
    current_load = None
    rate = None

    def __init__(self, rate=5000):
        self.rate = rate

    def more_allowed(self, client, just_checking=False):
        if self.window_start is None:
            if not just_checking:
                self.window_start = datetime.now()
                self.current_load = 1
            return True, 3600
        else:
            if datetime.now() - self.window_start > ONE_HOUR:
                self.window_start = datetime.now()
                self.current_load = 0

        remaining = self.rate
        if hasattr(client, 'rate_remaining'):
            remaining = client.rate_remaining
        allowed = self.current_load < self.rate and remaining > 0
        if not just_checking and allowed:
            self.current_load += 1
        waiting = (self.window_start + ONE_HOUR) - datetime.now()
        return allowed, waiting.total_seconds()
