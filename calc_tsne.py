#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""
simple Python wrapper for the bh_tsne binary.
adapted from https://github.com/ninjin/barnes-hut-sne
to deal directly with numpy array as input and output
"""

# Copyright (c) 2014, Géraud Le Falher <daureg@gmail.com>
# Copyright (c) 2013, Pontus Stenetorp <pontus stenetorp se>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import os.path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import numpy as np

# Constants
BH_TSNE_BIN_PATH = os.path.join(os.path.dirname(__file__), 'bh_tsne')
NOT_FOUND = 'Unable to find the bh_tsne binary in {}'.format(BH_TSNE_BIN_PATH)
# Default hyper-parameter values from van der Maaten (2013)
DEFAULT_PERPLEXITY = 30.0
DEFAULT_THETA = 0.5
###


class tSNE(object):
    """Mock Scikit manifold interface (at least the part needed)"""
    def __init__(self, n_components, theta=DEFAULT_THETA,
                 perplexity=DEFAULT_PERPLEXITY):
        """`theta` control approximation quality, 0 meaning exact algorithm"""
        self.n_components = n_components
        self.theta = float(theta)
        self.perplexity = float(perplexity)

    def fit_transform(self, data):
        return bh_tsne(data, self.perplexity, self.theta, self.n_components,
                       True)


class TmpDir(object):
    def __enter__(self):
        self._tmp_dir_path = tempfile.mkdtemp()
        return self._tmp_dir_path

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self._tmp_dir_path)


def _read_unpack(fmt, fh):
    return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))


def bh_tsne(samples, perplexity=DEFAULT_PERPLEXITY, theta=DEFAULT_THETA,
            out_dims=2, verbose=False):
    assert os.path.isfile(BH_TSNE_BIN_PATH), NOT_FOUND
    sample_count, sample_dim = samples.shape

    # bh_tsne works with fixed input and output paths, give it a temporary
    # directory to work in so we don't clutter the filesystem
    with TmpDir() as tmp_dir_path:
        # Note: The binary format used by bh_tsne is roughly the same as for
        # vanilla tsne
        with open(os.path.join(tmp_dir_path, 'data.dat'), 'wb') as data_file:
            # Write the bh_tsne header
            data_file.write(struct.pack('iidd', sample_count, sample_dim,
                                        theta, perplexity))
            # Then write the data
            for sample in samples:
                data_file.write(struct.pack('{}d'.format(len(sample)),
                                            *sample))

        if verbose:
            shutil.copy(os.path.join(tmp_dir_path, 'data.dat'),
                        '/home/lefag/thesis/illalla')
        # Call bh_tsne and let it do its thing
        with open('/dev/null', 'w') as dev_null:
            # bh_tsne is very noisy on stdout, tell it to use stderr if it is
            # to print any output
            output = sys.stdout if verbose else dev_null
            binary = os.path.abspath(BH_TSNE_BIN_PATH)
            args = shlex.split('{} {}'.format(binary, out_dims))
            try:
                subprocess.check_call(args, cwd=tmp_dir_path, stdout=output)
            except subprocess.CalledProcessError:
                if not verbose:
                    print('Enable verbose mode for more details')
                raise

        # Read and pass on the results
        with open(os.path.join(tmp_dir_path, 'result.dat'), 'rb') as output:
            # The first two integers are just the number of samples and the
            # dimensionality
            result_samples, result_dims = _read_unpack('ii', output)
            # Collect the results, but they may be out of order
            results = [_read_unpack('{}d'.format(result_dims), output)
                       for _ in xrange(result_samples)]
            # Now collect the landmark data so that we can return the data in
            # the order it arrived
            results = sorted([(_read_unpack('i', output), e)
                              for e in results])
            ordered = np.zeros((result_samples, result_dims))
            for idx, point in results:
                ordered[idx[0], :] = point
            return ordered
            # The last piece of data is the cost for each sample, we ignore it
            # read_unpack('{}d'.format(sample_count), output_file)


if __name__ == '__main__':
    data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.5]])
    print(data)
    print(bh_tsne(data, out_dims=3, perplexity=0.1, verbose=True))
