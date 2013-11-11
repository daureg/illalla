#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Implement exact algorithm of the paper: Spatial Scan Statistics:
Approximations and Performance Study."""
import heapq
import numpy as np
from math import log


def get_discrepancy_function(total_m, total_b, support):
    """Return a binary function computing the Kulldorff discrepancy, given
    that there is total_m measured data, total_b background data and we want
    at least support points (otherwise return None)."""
    def discrepancy(m, b):
        """Compute d(m, b) and return None if it lacks support."""
        if m < support or b < support:
            return None
        m_ratio = m/total_m
        b_ratio = b/total_b
        return m_ratio*log(m_ratio/b_ratio) + \
            (1 - m_ratio)*log((1-m_ratio)/(1-b_ratio))
    return discrepancy


def add_maybe(new_value, values_so_far, max_nb_values):
    """Consider adding new_value to values_so_far if it is one of the largest
    max_nb_values."""
    real_value, info = new_value
    if real_value is None:
        return values_so_far
    if len(values_so_far) < max_nb_values:
        heapq.heappush(values_so_far, new_value)
    else:
        if real_value > values_so_far[0][0]:
            heapq.heappushpop(values_so_far, new_value)
    return values_so_far


def exact_grid(measured, background, discrepancy, nb_loc=5):
    """Given the two g×g arrays representing the measure of interest and the
    background data, find the nb_loc region that have the most discrepancy
    according to the provided binary function to compute it."""
    assert np.size(measured) == np.size(background), "use same size input"
    grid_size = np.size(measured, 0)
    max_values = []
    for i in range(grid_size):  # left line
        cum_m = np.cumsum(measured[:, i])
        cum_b = np.cumsum(background[:, i])
        for j in range(1, grid_size):  # right line
            m = 0
            b = 0
            for y in range(grid_size):
                m += measured(j, y)
                b += measured(j, y)
                cum_m[y] += m
                cum_b[y] += b
                for k in range(grid_size):  # bottom line
                    for l in range(k, grid_size):  # top line
                        if k == 1:
                            m = cum_m[k]
                            b = cum_b[k]
                        else:
                            m = cum_m[l] - cum_m[k-1]
                            b = cum_b[l] - cum_b[k-1]
                            max_values = add_maybe((discrepancy(m, b),
                                                    ((i, k), (j, l))),
                                                   max_values, nb_loc)
    return max_values
