#

import brassboard_seq_test_utils as test_utils

import itertools
import pytest
import random

_ranges = set()
_starts = set()
_ends = set()

_add_ranges = _ranges.add
_add_starts = _starts.add
_add_ends = _ends.add

def check_max_range(vs):
    n = len(vs)
    _ranges.clear()
    for mr in test_utils.get_max_range(vs):
        assert mr not in _ranges
        _add_ranges(mr)
        (i0, i1, maxv) = mr
        _add_starts(i0)
        _add_ends(i1)
        assert test_utils.check_range(vs, i0, i1, maxv)
        assert i0 == 0 or vs[i0 - 1] < maxv
        assert i1 == n - 1 or vs[i1 + 1] < maxv
    if n == 0:
        return
    assert 0 in _starts
    assert n - 1 in _ends
    for i in range(1, n - 1):
        v = vs[i]
        if v > vs[i - 1]:
            assert i in _starts
        if v > vs[i + 1]:
            assert i in _ends

def test_0():
    assert test_utils.get_max_range([]) == []

@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5, 6])
def test_short(n):
    for l in itertools.product(*(range(0, n + 2),) * n):
        check_max_range(list(l))

@pytest.mark.parametrize('m,n', [(100, 1000), (1000, 300), (3000, 100)])
def test_random(m, n):
    for _ in range(n):
        l = []
        for i in range(m):
            l.append(random.randint(-10000, 10000))
        check_max_range(l)
