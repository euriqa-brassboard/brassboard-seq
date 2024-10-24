#

import brassboard_seq_test_utils as test_utils

import itertools
import pytest
import random

def check_suffix_array(l):
    assert l[-1] == 0
    res = test_utils.get_suffix_array(l)
    height = test_utils.get_height_array(l, res)
    assert res[0] == len(l) - 1
    prev_suffix = l[res[0]:]
    for i in range(1, len(l)):
        suffix = l[res[i]:]
        assert prev_suffix < suffix
        h = 0 if i == 1 else height[i - 2]
        assert len(prev_suffix) >= h
        assert len(suffix) >= h
        assert suffix[:h] == prev_suffix[:h]
        assert len(prev_suffix) == h or len(suffix) == h or prev_suffix[h] != suffix[h]
        prev_suffix = suffix
    assert sorted(res) == list(range(len(l)))

def test_0():
    assert test_utils.get_suffix_array([]) == []

@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5, 6, 7])
def test_short(n):
    for l in itertools.product(*(range(1, n + 1),) * n):
        check_suffix_array([*l, 0])

@pytest.mark.parametrize('m,n', [(100, 1000), (1000, 300), (3000, 100)])
def test_random(m, n):
    for _ in range(n):
        l = []
        for i in range(m):
            l.append(random.randint(1, m))
        l.append(0)
        check_suffix_array(l)
