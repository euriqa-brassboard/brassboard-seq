#

from brassboard_seq import scan

import pytest

def test_parampack():
    p = scan.ParamPack()
    assert scan.get_visited(p) is False

    assert scan.get_param(p) is p
    assert isinstance(scan.get_param(None), scan.ParamPack)

    assert str(p) == '<Undefined>'
    assert not 'a' in p
    assert str(p) == '<Undefined>'

    assert isinstance(p.a, scan.ParamPack)
    assert isinstance(p.b, scan.ParamPack)

    assert str(p) == '{}'
    assert repr(p) == '{}'
    assert 'a' not in p
    assert 'b' not in p

    assert p.a(1) == 1
    assert 'a' in p
    with pytest.raises(TypeError, match="Scalar value"):
        assert 'b' not in p.a
    assert 'b' not in p
    assert scan.get_visited(p) == dict(a=True)
    assert scan.get_visited(p.a) is True
    assert p.a() == 1
    assert isinstance(p.b(dict(x=2), a=2.1, c=3), scan.ParamPack)
    assert scan.get_visited(p.b) == {}
    assert scan.get_visited(p.b.x) is False
    assert scan.get_visited(p.b.a) is False
    assert scan.get_visited(p.b.c) is False
    assert scan.get_visited(p.b.undefined) is False
    assert p.b.x() == 2
    assert p.b.a() == 2.1
    assert p.b.c() == 3
    assert p.b.c(10) == 3
    assert scan.get_visited(p.b) == dict(x=True, a=True, c=True)
    assert scan.get_visited(p.b.x) is True
    assert scan.get_visited(p.b.a) is True
    assert scan.get_visited(p.b.c) is True
    assert scan.get_visited(p.b.undefined) is False

    with pytest.raises(TypeError, match="Cannot access"):
        p.a(b=1)

    with pytest.raises(TypeError, match="Cannot override"):
        p(dict(a=dict(b=1)))

    with pytest.raises(TypeError, match="Cannot get"):
        p.b(2)

    with pytest.raises(TypeError, match="Cannot override"):
        p(dict(b=2))

    with pytest.raises(TypeError, match="Cannot use"):
        p.b(10, dict(x=2), a=2.1, c=3)

    with pytest.raises(TypeError, match="Cannot get"):
        p.b()

    with pytest.raises(KeyError, match="Value is not assigned"):
        p.c()

    with pytest.raises(AttributeError):
        p._a

    with pytest.raises(AttributeError):
        p._a = 2

    p.c(dict(x=2), x=10)
    assert scan.get_visited(p.c) == {}
    assert p.c.x() == 2
    assert scan.get_visited(p.c) == dict(x=True)

    p.c.x = 3.4
    assert p.c.x() == 3.4

    p.c = dict(y=3)
    assert scan.get_visited(p.c) == dict(x=True)
    assert str(p.c) == 'x: 3.4\ny: 3'
    assert scan.get_visited(p.c) == dict(x=True)
    assert str(p.c.y) == "3"
    assert p.c.y() == 3
    assert scan.get_visited(p.c) == dict(x=True, y=True)

    with pytest.raises(TypeError, match="Cannot override"):
        p.c = 1
    with pytest.raises(TypeError, match="Cannot override"):
        p.c.y = {}

    p.c = dict(y=4)
    assert p.c.y() == 4
    assert str(p.c) == 'x: 3.4\ny: 4'
    p.c.z = dict(y=4, x=3, d=dict(a=2, b=3))
    assert scan.get_visited(p.c) == dict(x=True, y=True)
    assert p.c.z.y() == 4
    assert scan.get_visited(p.c) == dict(x=True, y=True, z=dict(y=True))
    assert p.c.z.x() == 3
    assert scan.get_visited(p.c) == dict(x=True, y=True, z=dict(y=True, x=True))
    assert p.c.z.d.a() == 2
    assert scan.get_visited(p.c) == dict(x=True, y=True,
                                         z=dict(y=True, x=True, d=dict(a=True)))
    assert p.c.z.d.b() == 3
    assert scan.get_visited(p.c) == dict(x=True, y=True,
                                         z=dict(y=True, x=True,
                                                d=dict(a=True, b=True)))
    p.c.z = dict(y=2, x=5, f=dict(a=1, b=2.3), d=dict(c=2, b=0.1))
    assert scan.get_visited(p.c) == dict(x=True, y=True,
                                         z=dict(y=True, x=True,
                                                d=dict(a=True, b=True)))
    assert p.c.z.y() == 2
    assert p.c.z.x() == 5
    assert p.c.z.d.a() == 2
    assert p.c.z.d.b() == 0.1
    assert scan.get_visited(p.c) == dict(x=True, y=True,
                                         z=dict(y=True, x=True,
                                                d=dict(a=True, b=True)))
    assert p.c.z.d.c() == 2
    assert scan.get_visited(p.c) == dict(x=True, y=True,
                                         z=dict(y=True, x=True,
                                                d=dict(a=True, b=True, c=True)))
    assert p.c.z.f.a() == 1
    assert scan.get_visited(p.c) == dict(x=True, y=True,
                                         z=dict(y=True, x=True,
                                                d=dict(a=True, b=True, c=True),
                                                f=dict(a=True)))
    assert p.c.z.f.b() == 2.3
    assert scan.get_visited(p.c) == dict(x=True, y=True,
                                         z=dict(y=True, x=True,
                                                d=dict(a=True, b=True, c=True),
                                                f=dict(a=True, b=True)))


def test_constructor():
    with pytest.raises(TypeError,
                       match="Cannot use value as default value for parameter pack"):
        scan.ParamPack(1)
    assert str(scan.ParamPack({})) == '{}'
    assert str(scan.ParamPack({'A': 123})) == 'A: 123'
    assert str(scan.ParamPack(B=23)) == 'B: 23'


def test_getitem():
    p = scan.ParamPack()
    d = p[:]
    assert d == {}
    assert d is not p[:]
    with pytest.raises(ValueError, match="Invalid index for ParamPack: 1"):
        p[1]

    p.a = 2
    assert d == {}
    assert p[:] == {"a": 2}

    p.b = {}
    d2 = p[:]
    assert d2 == {"a": 2, "b": {}}
    d2["b"]["c"] = 2
    assert p[:] == {"a": 2, "b": {}}
    p.b.c = 3
    assert p[:] == {"a": 2, "b": {"c": 3}}
    assert d2["b"]["c"] == 2
