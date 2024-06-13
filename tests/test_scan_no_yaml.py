#

from brassboard_seq import scan

import pytest

import types
import sys

yaml = types.ModuleType('yaml')
sys.modules['yaml'] = yaml

def test_parampack():
    p = scan.ParamPack()
    assert scan.get_visited(p) is False

    assert str(p) == '<Undefined>'

    assert isinstance(p.a, scan.ParamPack)
    assert isinstance(p.b, scan.ParamPack)

    assert str(p) == '{}'
    assert repr(p) == '{}'

    assert p.a(1) == 1
    assert scan.get_visited(p) == dict(a=True)
    assert scan.get_visited(p.a) is True
    assert p.a() == 1
    assert isinstance(p.b(dict(x=2), a=2.1, c=3), scan.ParamPack)
    assert p.b.x() == 2
    assert p.b.a() == 2.1
    assert p.b.c() == 3
    assert p.b.c(10) == 3

    p.c(dict(x=2), x=10)
    assert p.c.x() == 2

    p.c.x = 3.4
    assert p.c.x() == 3.4

    p.c = dict(y=3)
    assert str(p.c) == "{'x': 3.4, 'y': 3}"
    assert str(p.c.y) == "3"
    assert p.c.y() == 3

    p.c = dict(y=4)
    assert p.c.y() == 4
    assert str(p.c) == "{'x': 3.4, 'y': 4}"
