#

from brassboard_seq.rtval import new_extern, get_value
from numpy import sin

cb0 = lambda: 123
cb1 = lambda: 1.2

def f(n):
    v1 = new_extern(cb0)
    v2 = new_extern(cb1)
    v3 = 5 * sin(v1 + v2) - 10
    v = v3 - v1
    for i in range(n):
        get_value(v, i)

for i in range(1000):
    f(10_000)
