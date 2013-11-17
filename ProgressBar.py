#! /usr/bin/python2
# vim: set fileencoding=utf-8
import sys
import time
from timeit import default_timer as clock


# from http://thelivingpearl.com/progress_bar_1-py-code/
# add unicode and time remaining
class ProgressBar(object):
    """ProgressBar class holds the options of the progress bar.
    The options are:
        start   State from which start the progress. For example, if start is
                5 and the end is 10, the progress of this state is 50%
        end     State in which the progress has terminated.
        width   --
        fill    String to use for "filled" used to represent the progress
        blank   String to use for "filled" used to represent remaining space.
        format  Format
        incremental
    """
    def __init__(self, start=0, end=10, width=12, fill=u'█', blank=' ',
                 format=u'▏{fill}{blank}▕ {progress}% ETA: {eta:.1f}s',
                 incremental=True):
        super(ProgressBar, self).__init__()

        self.start = start
        self.end = end
        self.width = width
        self.fill = fill
        self.blank = blank
        self.format = format
        self.incremental = incremental
        self.step = 100 / float(width)  # fix
        self.sof = None
        self.remaining = -1.0
        self.reset()

    def __add__(self, increment):
        if self.sof is None:
            self.sof = clock()
            elapsed = 0
        else:
            elapsed = clock() - self.sof
        increment = self._get_progress(increment)
        if 100 > self.progress + increment:
            self.progress += increment
            self.remaining = 100*elapsed/self.progress - elapsed
        else:
            self.progress = 100
            self.remaining = 0
        return self

    def __unicode__(self):
        progressed = int(self.progress / self.step)  # fix
        fill = progressed * self.fill
        blank = (self.width - progressed) * self.blank
        return self.format.format(fill=fill, blank=blank,
                                  progress=int(self.progress),
                                  eta=self.remaining)

    def __str__(self):
        return unicode(self).encode('utf-8')

    __repr__ = __str__

    def _get_progress(self, increment):
        return float(increment * 100) / self.end

    def reset(self):
        """Resets the current progress to the start point"""
        self.progress = self._get_progress(self.start)
        return self


class AnimatedProgressBar(ProgressBar):
    """Extends ProgressBar to allow you to use it straighforward on a script.
    Accepts an extra keyword argument named `stdout` (by default use
    sys.stdout) and may be any file-object to which send the progress status.
    """
    def __init__(self, *args, **kwargs):
        super(AnimatedProgressBar, self).__init__(*args, **kwargs)
        self.stdout = kwargs.get('stdout', sys.stdout)

    def show_progress(self):
        if hasattr(self.stdout, 'isatty') and self.stdout.isatty():
            self.stdout.write('\r')
        else:
            self.stdout.write('\n')
        self.stdout.write(str(self))
        self.stdout.flush()

if __name__ == '__main__':
    p = AnimatedProgressBar(end=200, width=80, blank=' ', fill=u'█')

    while True:
        p + 5
        p.show_progress()
        time.sleep(0.2)
        if p.progress == 100:
            break
    print  # new line
