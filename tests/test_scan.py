#

from brassboard_seq import scan

import numpy as np
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
    with pytest.raises(TypeError, match="Cannot access value as parameter pack."):
        p.a[:]

    p.b = {}
    d2 = p[:]
    assert d2 == {"a": 2, "b": {}}
    d2["b"]["c"] = 2
    assert p[:] == {"a": 2, "b": {}}
    p.b.c = 3
    assert p[:] == {"a": 2, "b": {"c": 3}}
    assert d2["b"]["c"] == 2

def test_scan():
    sg = scan.ScanGroup()
    assert str(sg) == "ScanGroup\n"
    assert str(sg) == repr(sg)
    assert sg.groupsize() == 1
    assert sg.scansize(0) == 1
    assert sg.scandim(0) == 0
    assert sg.axisnum(0, 0) == 0
    assert sg.axisnum(0, -1) == 0
    assert sg.axisnum(0, 1) == 0
    assert sg.nseq() == 1
    assert sg.getbaseidx(0) == -1

    with pytest.raises(IndexError, match="Scan group index out of bound: -1."):
        sg.scansize(-1)
    with pytest.raises(IndexError, match="Scan group index out of bound: 1."):
        sg.scansize(1)
    with pytest.raises(IndexError, match="Scan group index out of bound: -1."):
        sg.scandim(-1)
    with pytest.raises(IndexError, match="Scan group index out of bound: 1."):
        sg.scandim(1)
    with pytest.raises(IndexError, match="Scan group index out of bound: -1."):
        sg.axisnum(-1, 0)
    with pytest.raises(IndexError, match="Scan group index out of bound: 1."):
        sg.axisnum(1, 0)
    with pytest.raises(IndexError, match="Scan group index out of bound: 1."):
        sg.getbaseidx(1)
    with pytest.raises(IndexError, match="Scan group index out of bound: -1."):
        sg.getbaseidx(-1)
    with pytest.raises(IndexError, match="Scan group index out of bound: -1."):
        sg[-2]
    with pytest.raises(IndexError, match="Scan group index out of bound: -1."):
        sg[-2] = {}
    with pytest.raises(TypeError,
                       match=f"Invalid type {list} in scan assignment."):
        sg[0] = []
    with pytest.raises(IndexError, match="Scan group index out of bound: -1."):
        sg.get_single_axis(-1)
    with pytest.raises(IndexError, match="Scan group index out of bound: 1."):
        sg.get_single_axis(1)
    with pytest.raises(IndexError,
                       match=f"Cannot set base to non-existing scan: 3."):
        sg.setbaseidx(0, 3)
    with pytest.raises(IndexError,
                       match=f"Invalid base index: -2."):
        sg.setbaseidx(0, -2)
    sg.setbaseidx(0, -1)

    assert str(sg) == "ScanGroup\n"
    assert str(sg) == repr(sg)
    assert str(sg[:]) == "Scan Base:\n  <empty>\n"
    assert str(sg[:]) == repr(sg[:])
    assert str(sg[0]) == "Scan 0:\n  <empty>\n"
    assert str(sg[0]) == repr(sg[0])
    assert str(sg[-1]) == str(sg[0])
    assert sg.groupsize() == 1
    assert sg.scansize(0) == 1
    assert sg.scandim(0) == 0
    assert sg.axisnum(0, 0) == 0
    assert sg.axisnum(0, -1) == 0
    assert sg.axisnum(0, 1) == 0
    assert sg.nseq() == 1
    assert sg.getseq_in_scan(0, 0) == {}
    assert sg.getseq(0) == {}
    with pytest.raises(IndexError, match="Sequence index out of bound: -1"):
        sg.getseq(-1)
    with pytest.raises(IndexError, match="Sequence index out of bound: 1"):
        sg.getseq(1)
    with pytest.raises(IndexError, match="Sequence index out of bound: -1"):
        sg.getseq_in_scan(0, -1)
    with pytest.raises(IndexError, match="Sequence index out of bound: 1"):
        sg.getseq_in_scan(0, 1)
    with pytest.raises(IndexError, match="Scan group index out of bound: -1"):
        sg.getseq_in_scan(-1, 0)
    with pytest.raises(IndexError, match="Scan group index out of bound: 1"):
        sg.getseq_in_scan(1, 0)

    with pytest.raises(AttributeError,
                       match="'brassboard_seq.scan.ScanWrapper' object has no attribute '_a'"):
        sg[:]._a

    with pytest.raises(AttributeError,
                       match="'brassboard_seq.scan.ScanWrapper' object has no attribute '_b'"):
        sg[:]._b = 2

    with pytest.raises(SyntaxError, match="Invalid scan syntax"):
        sg[0].scan[0] = [1, 2]
    with pytest.raises(SyntaxError, match="Invalid scan syntax"):
        sg[0].a.b[0] = [1, 2]
    with pytest.raises(IndexError, match="Scan dimension must not be negative: -1"):
        sg[0].a.scan[-1] = [1, 2]
    with pytest.raises(SyntaxError, match="Invalid scan syntax"):
        sg[0].scan([1, 2, 3])
    with pytest.raises(SyntaxError, match="Invalid scan syntax"):
        sg[0].a.b([1, 2, 3])
    with pytest.raises(TypeError,
                       match="Scan syntax takes 1 or 2 arguments, but 0 were given"):
        sg[0].a.scan()
    with pytest.raises(TypeError,
                       match="Scan syntax takes 1 or 2 arguments, but 3 were given"):
        sg[0].a.scan(1, 2, 3)
    with pytest.raises(IndexError,
                       match="Scan dimension must not be negative: -2."):
        sg[0].a.scan(-2, [2])

    sg.new_empty()
    assert sg.groupsize() == 1
    assert sg.scansize(0) == 1
    assert sg.scandim(0) == 0
    assert sg.axisnum(0, 0) == 0
    assert sg.axisnum(0, -1) == 0
    assert sg.axisnum(0, 1) == 0
    assert sg.nseq() == 1

    sg.new_empty()
    assert sg.groupsize() == 2
    assert sg.scansize(0) == 1
    assert sg.scansize(1) == 1
    assert sg.scandim(0) == 0
    assert sg.scandim(1) == 0
    assert sg.axisnum(0, 0) == 0
    assert sg.axisnum(0, -1) == 0
    assert sg.axisnum(0, 1) == 0
    assert sg.axisnum(1, 0) == 0
    assert sg.axisnum(1, -1) == 0
    assert sg.axisnum(1, 1) == 0
    assert sg.nseq() == 2
    assert str(sg[1]) == "Scan 1:\n  <empty>\n"
    assert str(sg[1]) == repr(sg[1])
    assert sg.groupsize() == 2
    assert sg.nseq() == 2

    assert str(sg[2]) == "Scan 2:\n  <empty>\n"
    assert str(sg[2]) == repr(sg[2])
    assert sg.groupsize() == 3
    assert sg.nseq() == 3

    sg = scan.ScanGroup()
    sg[:].a = 1
    assert sg.groupsize() == 1
    assert sg.nseq() == 1
    sg.new_empty()
    assert sg.groupsize() == 1
    assert sg.nseq() == 1

    sg = scan.ScanGroup()
    sg[0].a = 1
    assert sg.groupsize() == 1
    assert sg.nseq() == 1
    sg.new_empty()
    assert sg.groupsize() == 2
    assert sg.nseq() == 2

    sg[:].a.scan([1, 2, 3])
    assert sg.groupsize() == 2
    assert sg.scansize(0) == 1
    assert sg.scansize(1) == 3
    assert sg.scandim(0) == 0
    assert sg.scandim(1) == 1
    assert sg.axisnum(0, 0) == 0
    assert sg.axisnum(0, -1) == 0
    assert sg.axisnum(0, 1) == 0
    assert sg.axisnum(1, 0) == 1
    assert sg.axisnum(1, -1) == 0
    assert sg.axisnum(1, 1) == 0
    assert sg.nseq() == 4

    sg[1].b.scan([3, 4, 5])
    assert sg.groupsize() == 2
    assert sg.scansize(0) == 1
    assert sg.scansize(1) == 3
    assert sg.scandim(0) == 0
    assert sg.scandim(1) == 1
    assert sg.axisnum(0, 0) == 0
    assert sg.axisnum(0, -1) == 0
    assert sg.axisnum(0, 1) == 0
    assert sg.axisnum(1, 0) == 2
    assert sg.axisnum(1, -1) == 0
    assert sg.axisnum(1, 1) == 0
    assert sg.nseq() == 4
    assert sg.getseq_in_scan(0, 0) == dict(a=1)
    assert sg.getseq_in_scan(1, 0) == dict(a=1, b=3)
    assert sg.getseq_in_scan(1, 1) == dict(a=2, b=4)
    assert sg.getseq_in_scan(1, 2) == dict(a=3, b=5)
    assert sg.getseq(0) == dict(a=1)
    assert sg.getseq(1) == dict(a=1, b=3)
    assert sg.getseq(2) == dict(a=2, b=4)
    assert sg.getseq(3) == dict(a=3, b=5)

    assert sg.get_fixed(0) == dict(a=1)
    assert sg.get_fixed(1) == {}
    assert sg.get_vars(1) == (dict(a=[1, 2, 3], b=[3, 4, 5]), 3)
    assert sg.get_vars(1, 0) == (dict(a=[1, 2, 3], b=[3, 4, 5]), 3)

    sgd = sg.dump()
    assert sgd == dict(version=1,
                       base=dict(params={},
                                 vars=[dict(size=3, params=dict(a=[1, 2, 3]))]),
                       scans=[dict(baseidx=0, params=dict(a=1), vars=[]),
                              dict(baseidx=0, params={},
                                   vars=[dict(size=3, params=dict(b=[3, 4, 5]))])])
    sg2 = scan.ScanGroup.load(sgd)
    assert str(sg) == str(sg2)
    assert sg2.dump() == sgd

    with pytest.raises(ValueError, match="Version missing."):
        scan.ScanGroup.load({})
    with pytest.raises(ValueError, match="Unsupported version: 0"):
        scan.ScanGroup.load(dict(version=0))

    with pytest.raises(IndexError, match="Scan dimension out of bound: 0."):
        sg.get_vars(0)
    with pytest.raises(IndexError, match="Scan dimension out of bound: -1."):
        sg.get_vars(1, -1)
    with pytest.raises(IndexError, match="Scan dimension out of bound: 1."):
        sg.get_vars(1, 1)

    assert str(sg) == """ScanGroup
  Scan Base:
    Scan dimension 0: (size 3)
       a: [1, 2, 3]
  Scan 0:
    Fixed parameters:
       a: 1
  Scan 1:
    Scan dimension 0: (size 3)
       b: [3, 4, 5]
"""
    assert str(sg[:]) == """Scan Base:
  Scan dimension 0: (size 3)
     a: [1, 2, 3]
"""
    assert str(sg[0]) == """Scan 0:
  Fixed parameters:
     a: 1
"""
    assert str(sg[1]) == """Scan 1:
  Scan dimension 0: (size 3)
     b: [3, 4, 5]
"""
    assert str(sg[:].a) == """Scan Base [.a]:
  Scan dimension 0: (size 3)
     [1, 2, 3]
"""
    assert str(sg[0].a) == """Scan 0 [.a]:
  Fixed parameters:
     1
"""
    assert str(sg[1].a) == """Scan 1 [.a]:
  Scan dimension 0: (size 3)
     <empty>
"""
    assert str(sg[:].b) == """Scan Base [.b]:
  Scan dimension 0: (size 3)
     <empty>
"""
    assert str(sg[0].b) == """Scan 0 [.b]:
  <empty>
"""
    assert str(sg[1].b) == """Scan 1 [.b]:
  Scan dimension 0: (size 3)
     [3, 4, 5]
"""

    sg.setbaseidx(1, 0)
    assert str(sg) == """ScanGroup
  Scan Base:
    Scan dimension 0: (size 3)
       a: [1, 2, 3]
  Scan 0:
    Fixed parameters:
       a: 1
  Scan 1:
    Base index: 0
    Scan dimension 0: (size 3)
       b: [3, 4, 5]
"""
    assert sg.getseq_in_scan(0, 0) == dict(a=1)
    assert sg.getseq_in_scan(1, 0) == dict(a=1, b=3)
    assert sg.getseq_in_scan(1, 1) == dict(a=1, b=4)
    assert sg.getseq_in_scan(1, 2) == dict(a=1, b=5)
    assert sg.getseq(0) == dict(a=1)
    assert sg.getseq(1) == dict(a=1, b=3)
    assert sg.getseq(2) == dict(a=1, b=4)
    assert sg.getseq(3) == dict(a=1, b=5)

    assert str(sg2[0] + sg[1]) == """ScanGroup
  Scan 0:
    Fixed parameters:
       a: 1
  Scan 1:
    Fixed parameters:
       a: 1
    Scan dimension 0: (size 3)
       b: [3, 4, 5]
"""

    assert str(sg[0] + sg2[1]) == """ScanGroup
  Scan 0:
    Fixed parameters:
       a: 1
  Scan 1:
    Scan dimension 0: (size 3)
       b: [3, 4, 5]
       a: [1, 2, 3]
"""

    assert str(sg[0] + sg2) == """ScanGroup
  Scan 0:
    Fixed parameters:
       a: 1
  Scan 1:
    Fixed parameters:
       a: 1
  Scan 2:
    Scan dimension 0: (size 3)
       b: [3, 4, 5]
       a: [1, 2, 3]
"""

    assert str(sg + sg2[1]) == """ScanGroup
  Scan 0:
    Fixed parameters:
       a: 1
  Scan 1:
    Fixed parameters:
       a: 1
    Scan dimension 0: (size 3)
       b: [3, 4, 5]
  Scan 2:
    Scan dimension 0: (size 3)
       b: [3, 4, 5]
       a: [1, 2, 3]
"""

    assert str(sg + sg2) == """ScanGroup
  Scan 0:
    Fixed parameters:
       a: 1
  Scan 1:
    Fixed parameters:
       a: 1
    Scan dimension 0: (size 3)
       b: [3, 4, 5]
  Scan 2:
    Fixed parameters:
       a: 1
  Scan 3:
    Scan dimension 0: (size 3)
       b: [3, 4, 5]
       a: [1, 2, 3]
"""

    with pytest.raises(TypeError, match=f"Invalid type {int} in scan concatenation."):
        sg + 1
    with pytest.raises(TypeError, match=f"Invalid type {float} in scan concatenation."):
        2.0 + sg[:]
    with pytest.raises(ValueError, match=f"Only top-level Scan can be concatenated."):
        sg + sg[:].a

    sg = scan.ScanGroup()
    sg[:].a.b.scan([])
    assert str(sg) == """ScanGroup
"""

    sg[:].a.c.scan([1])
    assert str(sg) == """ScanGroup
  Scan Base:
    Fixed parameters:
       a.c: 1
"""
    assert sg.get_single_axis(0) == (None, ())

    sg[:].a.d.scan([1, 2])
    assert str(sg) == """ScanGroup
  Scan Base:
    Fixed parameters:
       a.c: 1
    Scan dimension 0: (size 2)
       a.d: [1, 2]
"""
    assert sg.get_single_axis(0) == ([1, 2], ('a', 'd'))

    sg[:].a.e.scan[0] = np.array([0.4, 0.5])
    assert str(sg) == """ScanGroup
  Scan Base:
    Fixed parameters:
       a.c: 1
    Scan dimension 0: (size 2)
       a:
         d: [1, 2]
         e: [0.4, 0.5]
"""
    assert sg.get_single_axis(0) == ([1, 2], ('a', 'd'))

    with pytest.raises(ValueError,
                       match="Cannot scan a parameter in multiple dimensions."):
        sg[:].a.e.scan(2, [1, 2, 3, 4, 5])

    sg[:].a.g.scan(2, [1, 2, 3, 4, 5])
    assert str(sg) == """ScanGroup
  Scan Base:
    Fixed parameters:
       a.c: 1
    Scan dimension 0: (size 2)
       a:
         d: [1, 2]
         e: [0.4, 0.5]
    Scan dimension 2: (size 5)
       a.g: [1, 2, 3, 4, 5]
"""
    assert sg.get_single_axis(0) == (None, ())

    assert sg.nseq() == 10
    assert sg.getseq(0) == dict(a=dict(c=1, d=1, e=0.4, g=1))
    assert sg.getseq(1) == dict(a=dict(c=1, d=2, e=0.5, g=1))
    assert sg.getseq(2) == dict(a=dict(c=1, d=1, e=0.4, g=2))
    assert sg.getseq(3) == dict(a=dict(c=1, d=2, e=0.5, g=2))
    assert sg.getseq(4) == dict(a=dict(c=1, d=1, e=0.4, g=3))
    assert sg.getseq(5) == dict(a=dict(c=1, d=2, e=0.5, g=3))
    assert sg.getseq(6) == dict(a=dict(c=1, d=1, e=0.4, g=4))
    assert sg.getseq(7) == dict(a=dict(c=1, d=2, e=0.5, g=4))
    assert sg.getseq(8) == dict(a=dict(c=1, d=1, e=0.4, g=5))
    assert sg.getseq(9) == dict(a=dict(c=1, d=2, e=0.5, g=5))

    sg2 = scan.ScanGroup.load(sg.dump())
    assert sg2.dump() == sg.dump()
    assert str(sg2) == str(sg)

    sg2 = scan.ScanGroup.load(dict(version=1))
    assert str(sg2) == "ScanGroup\n"
    assert sg2.groupsize() == 1

    with pytest.raises(ValueError,
                       match="ScanGroup mismatch in assignment."):
        sg2[:] = sg[0]

    with pytest.raises(ValueError,
                       match="Invalid serialization of ScanGroup: empty scans array."):
        scan.ScanGroup.load(dict(version=1, scans=[]))

    sg2 = scan.ScanGroup.load(dict(version=1, base={}))
    assert str(sg2) == "ScanGroup\n"
    assert sg2.groupsize() == 1

    sg2 = scan.ScanGroup.load(dict(version=1,
                                   base=dict(vars=[dict(),
                                                   dict(size=3,
                                                        params=dict(a=[1, 2, 3]))])))
    assert str(sg2) == """ScanGroup
  Scan Base:
    Scan dimension 1: (size 3)
       a: [1, 2, 3]
"""
    assert sg2.groupsize() == 1
    assert sg2.nseq() == 3
    assert sg2.get_single_axis(0) == ([1, 2, 3], ('a',))

    with pytest.raises(ValueError,
                       match="Invalid serialization of ScanGroup: scan size mismatch, expect 3, got 2"):
        scan.ScanGroup.load(dict(version=1,
                                 base=dict(vars=[dict(size=3,
                                                      params=dict(a=[1, 2]))])))
    with pytest.raises(TypeError,
                       match=f"Invalid serialization of ScanGroup: wrong parameter type {int}"):
        scan.ScanGroup.load(dict(version=1,
                                 base=dict(vars=[dict(params=dict(a=1))])))

    sg = scan.ScanGroup()
    sg.setbaseidx(1, 0)
    assert sg.getbaseidx(0) == -1
    assert sg.getbaseidx(1) == 0
    sg.setbaseidx(1, -1)
    assert sg.getbaseidx(0) == -1
    assert sg.getbaseidx(1) == -1
    sg.setbaseidx(0, 1)
    assert sg.getbaseidx(0) == 1
    assert sg.getbaseidx(1) == -1
    with pytest.raises(ValueError,
                       match="Base index loop detected."):
        sg.setbaseidx(1, 0)

    sg = scan.ScanGroup()
    sg[:].a.b.scan[0] = [1, 2, 3]

    with pytest.raises(ValueError,
                       match="Scan parameter size does not match."):
        sg[:].a.c.scan[0] = [1, 2]

    sg = scan.ScanGroup()
    sg[:].a.b.scan[0] = [1, 2, 3]
    sg[0].a.c.scan[0] = [1, 2]

    with pytest.raises(ValueError,
                       match="Scan parameter size does not match."):
        sg.nseq()

    sg = scan.ScanGroup()
    sg[:].a.b = 1
    sg[0].a.b.scan[0] = 2
    assert sg.nseq() == 1
    assert sg.getseq(0) == dict(a=dict(b=2))

    with pytest.raises(TypeError,
                       match="Scan parameter cannot be a dict."):
        sg[:].a.b.scan(2, dict())
    with pytest.raises(TypeError,
                       match="Scan parameter cannot be a scan."):
        sg[:].a.b.scan(2, sg)
    with pytest.raises(TypeError,
                       match="Scan parameter cannot be a scan."):
        sg[:].a.b.scan(2, sg[0])
    with pytest.raises(TypeError,
                       match="Scan parameter cannot be a scan."):
        sg[:].a.c = sg
    with pytest.raises(TypeError,
                       match="Scan parameter cannot be a scan."):
        sg[:].a.c = sg[0]

    sg = scan.ScanGroup()
    sg[:].x.y = np.True_
    sg[:].x.y = np.False_
    assert sg.get_fixed(0)['x']['y'] is False

    sg[0].y.z = np.int32(2)
    assert sg.get_fixed(0)['y']['z'] == 2
    assert isinstance(sg.get_fixed(0)['y']['z'], int)

    sg[0].y.k = np.array([0.1, 2.3])
    assert sg.get_fixed(0)['y']['k'] == [0.1, 2.3]
    assert isinstance(sg.get_fixed(0)['y']['k'], list)

    sg[0].y.x = [np.float64(0.1), 2.3]
    assert sg.get_fixed(0)['y']['x'] == [0.1, 2.3]
    assert isinstance(sg.get_fixed(0)['y']['x'][0], float)

    sg[0].a.b = 1

    with pytest.raises(TypeError, match="Override dict not allowed."):
        sg[0].a = dict(x=2, y=3)

    with pytest.raises(TypeError, match="Scan parameter cannot be a dict."):
        sg[:].a.scan([dict(x=[2, 3], y=[3, 4]), 2])

    sg[:].a = dict(x=2, y=3)
    assert sg.get_fixed(0)['a'] == dict(b=1, x=2, y=3)

    assert str(sg) == """ScanGroup
  Scan Base:
    Fixed parameters:
       x.y: false
       a:
         x: 2
         y: 3
  Scan 0:
    Fixed parameters:
       y:
         z: 2
         k: [0.1, 2.3]
         x: [0.1, 2.3]
       a.b: 1
"""

    with pytest.raises(TypeError, match="Assignment to field of scalar not allowed."):
        sg[0].y.z.x = 2
    with pytest.raises(TypeError,
                       match="Changing field from non-dict to dict not allowed."):
        sg[0].y.z = {}
    with pytest.raises(TypeError,
                       match="Changing field from dict to non-dict not allowed."):
        sg[0].y = 2
    with pytest.raises(TypeError, match="Override dict not allowed."):
        sg[0].y = {}

    with pytest.raises(ValueError, match="Cannot scan a fixed parameter."):
        sg[0].y.z.scan([1, 2, 3])

    assert sg.getseq(0) == dict(x=dict(y=False), a=dict(x=2, y=3, b=1),
                                y=dict(z=2, k=[0.1, 2.3], x=[0.1, 2.3]))
    sg[:].y.z.x = 2
    assert sg.getseq(0) == dict(x=dict(y=False), a=dict(x=2, y=3, b=1),
                                y=dict(z=2, k=[0.1, 2.3], x=[0.1, 2.3]))
    sg[0].y.a.scan([1, 2, 3])
    with pytest.raises(ValueError, match="Cannot fix a scanned parameter."):
        sg[0].y.a = 2

    sg = scan.ScanGroup()
    sg[0].a.b = 1
    sg[0].a.c.scan[2] = [2, 3, 8]
    assert str(sg) == """ScanGroup
  Scan 0:
    Fixed parameters:
       a.b: 1
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
"""
    sg[1] = sg[0]
    assert str(sg) == """ScanGroup
  Scan 0:
    Fixed parameters:
       a.b: 1
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
  Scan 1:
    Fixed parameters:
       a.b: 1
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
"""
    sg.setbaseidx(1, 0)
    assert str(sg) == """ScanGroup
  Scan 0:
    Fixed parameters:
       a.b: 1
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
  Scan 1:
    Base index: 0
    Fixed parameters:
       a.b: 1
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
"""
    sg[0].d = 0
    sg[1].c = 2
    assert str(sg) == """ScanGroup
  Scan 0:
    Fixed parameters:
       a.b: 1
       d: 0
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
  Scan 1:
    Base index: 0
    Fixed parameters:
       a.b: 1
       c: 2
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
"""
    assert sg.nseq() == 6
    assert sg.getseq(0) == dict(a=dict(b=1, c=2), d=0)
    assert sg.getseq(1) == dict(a=dict(b=1, c=3), d=0)
    assert sg.getseq(2) == dict(a=dict(b=1, c=8), d=0)
    assert sg.getseq(3) == dict(a=dict(b=1, c=2), d=0, c=2)
    assert sg.getseq(4) == dict(a=dict(b=1, c=3), d=0, c=2)
    assert sg.getseq(5) == dict(a=dict(b=1, c=8), d=0, c=2)

    sg[3] = sg[1]
    assert str(sg) == """ScanGroup
  Scan 0:
    Fixed parameters:
       a.b: 1
       d: 0
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
  Scan 1:
    Base index: 0
    Fixed parameters:
       a.b: 1
       c: 2
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
  Scan 2:
    <empty>
  Scan 3:
    Base index: 0
    Fixed parameters:
       a.b: 1
       c: 2
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
"""
    sg[:] = sg[0]
    assert str(sg) == """ScanGroup
  Scan Base:
    Fixed parameters:
       a.b: 1
       d: 0
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
  Scan 0:
    Fixed parameters:
       a.b: 1
       d: 0
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
  Scan 1:
    Base index: 0
    Fixed parameters:
       a.b: 1
       c: 2
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
  Scan 2:
    <empty>
  Scan 3:
    Base index: 0
    Fixed parameters:
       a.b: 1
       c: 2
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
"""

    sg[:] = sg[1]
    assert str(sg) == """ScanGroup
  Scan Base:
    Fixed parameters:
       a.b: 1
       c: 2
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
  Scan 0:
    Fixed parameters:
       a.b: 1
       d: 0
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
  Scan 1:
    Base index: 0
    Fixed parameters:
       a.b: 1
       c: 2
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
  Scan 2:
    <empty>
  Scan 3:
    Base index: 0
    Fixed parameters:
       a.b: 1
       c: 2
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
"""

    sg[:] = sg[0]
    assert str(sg) == """ScanGroup
  Scan Base:
    Fixed parameters:
       a.b: 1
       d: 0
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
  Scan 0:
    Fixed parameters:
       a.b: 1
       d: 0
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
  Scan 1:
    Base index: 0
    Fixed parameters:
       a.b: 1
       c: 2
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
  Scan 2:
    <empty>
  Scan 3:
    Base index: 0
    Fixed parameters:
       a.b: 1
       c: 2
    Scan dimension 2: (size 3)
       a.c: [2, 3, 8]
"""
    assert str(sg[3].a) == """Scan 3 [.a]:
  Base index: 0
  Fixed parameters:
     b: 1
  Scan dimension 2: (size 3)
     c: [2, 3, 8]
"""
    assert str(sg[3].a.b) == """Scan 3 [.a.b]:
  Base index: 0
  Fixed parameters:
     1
  Scan dimension 2: (size 3)
     <empty>
"""
    assert str(sg[3].a.c) == """Scan 3 [.a.c]:
  Base index: 0
  Scan dimension 2: (size 3)
     [2, 3, 8]
"""
    assert str(sg[3].a.a.c) == """Scan 3 [.a.a.c]:
  Base index: 0
  Scan dimension 2: (size 3)
     <empty>
"""

    sg = scan.ScanGroup()
    sg[:].a.b = 1
    sg[0].a.b.c.d = 2
    assert sg.getseq(0) == dict(a=dict(b=dict(c=dict(d=2))))
