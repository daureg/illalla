#! /usr/bin/python2
# vim: set fileencoding=utf-8


# from https://stackoverflow.com/a/1336821
class Chunker(object):
    """Split `iterable` on evenly sized chunks.
    Leftovers are yielded at the end.
    """
    def __init__(self, chunksize):
        assert chunksize > 0
        self.chunksize = chunksize
        self.chunk = []

    def __call__(self, iterable):
        """Yield items from `iterable` `self.chunksize` at the time."""
        assert len(self.chunk) < self.chunksize
        for item in iterable:
            self.chunk.append(item)
            if len(self.chunk) == self.chunksize:
                yield self.chunk
                self.chunk = []

        if len(self.chunk) > 0:
            yield self.chunk

if __name__ == '__main__':
    chunker = Chunker(3)
    res = [''.join(chunk) for chunk in chunker('abcdefghij')]
    assert res == ['abc', 'def', 'ghi', 'j']
