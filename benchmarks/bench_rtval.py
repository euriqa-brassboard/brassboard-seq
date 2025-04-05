#

from brassboard_seq.rtval import get_value
from brassboard_seq_test_utils import new_extern
from numpy import sin

cb0 = lambda: 123
cb1 = lambda: 1.2

def f():
    v1 = new_extern(cb0)
    v2 = new_extern(cb1)
    v3 = 5 * sin(v1 + v2) - 10
    return get_value(v3 - v1, 1)

for i in range(10_000_000):
    f()
