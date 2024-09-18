#

from brassboard_seq import rtval
import brassboard_seq_test_utils as test_utils

import numpy as np
import math
import operator
import pytest

class CmpError:
    def __eq__(self, other):
        raise RuntimeError("Must not call")

def cmp_value(v1, v2):
    if np.isfinite(v1) and np.isfinite(v2):
        if isinstance(v2, np.float32):
            return v1 == pytest.approx(v2, rel=1e-4)
        return v1 == pytest.approx(v2)
    if math.isnan(v1):
        return math.isnan(v2)
    if math.isinf(v1):
        return math.isinf(v2) and v1 * v2 > 0
    return False

def f(v1, v2):
    v3 = v1 + v2 * 3
    v4 = np.sin(v3)
    return v1 * 2 + v4 + 10

def test_rtval():
    with pytest.raises(TypeError):
        rtval.RuntimeValue()

    v1 = rtval.new_extern(lambda: 1)
    s1 = str(v1)
    assert s1.startswith('extern(')
    v2 = rtval.new_extern_age(lambda age: 2)
    s2 = str(v2)
    assert s2.startswith('extern_age(')

    with pytest.raises(TypeError, match="Cannot convert runtime value to boolean"):
        bool(v1)
    with pytest.raises(TypeError, match="Cannot convert runtime value to boolean"):
        if v1:
            pass
    with pytest.raises(TypeError, match="Cannot convert runtime value to boolean"):
        assert v1

    with pytest.raises(TypeError):
        np.sum(v1)
    with pytest.raises(TypeError):
        np.sin.at(v1, 0)
    with pytest.raises(TypeError):
        np.conj(v1)

    c1 = rtval.new_const(1.2)
    c2 = rtval.new_const(2.3)
    assert str(c1) == '1.2'
    assert str(c2) == '2.3'
    c3 = c1 + c2
    assert isinstance(c3, rtval.RuntimeValue)
    assert str(c3) == '3.5'
    assert str(v1 + c1) == f'1.2 + {s1}'
    assert str(c2 + v2) == f'2.3 + {s2}'
    assert str((v1 + c1) + c2) == f'3.5 + {s1}'
    assert str(c1 + (c2 + v2)) == f'3.5 + {s2}'

    ce1 = CmpError()
    ce2 = CmpError()
    with pytest.raises(RuntimeError, match="Must not call"):
        ce1 == ce2
    assert not rtval.same_value(ce1, ce2)

    assert not rtval.same_value(v1, v2)
    assert rtval.same_value(v1, v1)
    assert rtval.same_value(v2, v2)
    assert rtval.same_value(v1, +v1)

    assert not rtval.same_value(1, v1)
    assert not rtval.same_value(v2, 3)

    assert rtval.same_value(1, 1)
    assert rtval.same_value(1, 1.0)
    assert not rtval.same_value(1, 0)

    assert not (v1 != v1)
    assert v2 == v2

    av1 = abs(v1)
    assert str(av1) == f'abs({s1})'
    assert abs(av1) is av1
    assert np.abs(av1) is av1

    cv1 = math.ceil(v1)
    assert str(cv1) == f'ceil({s1})'
    assert math.ceil(cv1) is cv1
    assert np.ceil(cv1) is cv1

    fv1 = math.floor(v1)
    assert str(fv1) == f'floor({s1})'
    assert math.floor(fv1) is fv1
    assert np.floor(fv1) is fv1

    rv1 = np.rint(v1)
    assert str(rv1) == f'rint({s1})'
    assert np.rint(rv1) is rv1

    iv1 = rtval.inv(v1)
    iv2 = np.logical_not(v1)
    assert str(iv1) == f'inv({s1})'
    assert str(iv2) == f'inv({s1})'
    assert str(np.logical_not(iv1)) == f'bool({s1})'
    assert str(rtval.inv(iv1)) == f'bool({s1})'
    assert str(np.logical_not(iv2)) == f'bool({s1})'
    assert str(rtval.inv(iv2)) == f'bool({s1})'
    assert rtval.convert_bool(iv1) is iv1
    assert rtval.convert_bool(iv2) is iv2

    r64v1 = rtval.round_int64(v1)
    assert str(r64v1) == f'int64({s1})'
    assert rtval.round_int64(r64v1) is r64v1
    assert round(r64v1) is r64v1
    assert str(rtval.convert_bool(r64v1)) == f'bool({s1})'

    bv1 = rtval.convert_bool(v1)
    assert str(bv1) == f'bool({s1})'
    assert rtval.convert_bool(bv1) is bv1

    assert np.fmin(v1, v1) is v1
    assert np.fmax(v2, v2) is v2
    assert str(np.fmin(v1, v2)) == f'min({s1}, {s2})'
    assert str(np.fmax(v2, v1)) == f'max({s2}, {s1})'

    assert rtval.ifelse(True, v1, v2) is v1
    assert rtval.ifelse(False, v1, v2) is v2
    assert str(rtval.ifelse(v1, 3, v2)) == f'ifelse({s1}, 3, {s2})'
    assert rtval.ifelse(v1, 3, 3) == 3
    assert rtval.ifelse(v1, v2, v2) is v2

    assert str(np.exp(v1)) == f'exp({s1})'
    assert str(np.expm1(v1)) == f'expm1({s1})'
    assert str(np.log(v1)) == f'log({s1})'
    assert str(np.log1p(v1)) == f'log1p({s1})'
    assert str(np.log2(v1)) == f'log2({s1})'
    assert str(np.log10(v1)) == f'log10({s1})'
    assert str(np.sqrt(v1)) == f'sqrt({s1})'
    assert str(np.arcsin(v1)) == f'arcsin({s1})'
    assert str(np.arccos(v1)) == f'arccos({s1})'
    assert str(np.arctan(v1)) == f'arctan({s1})'
    assert str(np.arcsinh(v1)) == f'arcsinh({s1})'
    assert str(np.arccosh(v1)) == f'arccosh({s1})'
    assert str(np.arctanh(v1)) == f'arctanh({s1})'
    assert str(np.sin(v1)) == f'sin({s1})'
    assert str(np.cos(v1)) == f'cos({s1})'
    assert str(np.tan(v1)) == f'tan({s1})'
    assert str(np.sinh(v1)) == f'sinh({s1})'
    assert str(np.cosh(v1)) == f'cosh({s1})'
    assert str(np.tanh(v1)) == f'tanh({s1})'
    assert str(np.arctan2(v1, v2)) == f'arctan2({s1}, {s2})'
    assert str(np.hypot(v1, v2)) == f'hypot({s1}, {s2})'

    npval = np.int32(3)
    assert str(npval + v1) == f'3 + {s1}'
    assert str(npval - v1) == f'3 - {s1}'
    assert str(npval * v1) == f'3 * {s1}'
    assert str(npval / v1) == f'3 / {s1}'
    assert str(npval % v1) == f'3 % {s1}'
    assert str(npval & v1) == f'3 & {s1}'
    assert str(npval | v1) == f'3 | {s1}'
    assert str(npval ^ v1) == f'3 ^ {s1}'
    assert str(npval ** v1) == f'3**{s1}'
    assert str(npval < v1) == f'3 < {s1}'
    assert str(npval > v1) == f'3 > {s1}'
    assert str(npval <= v1) == f'3 <= {s1}'
    assert str(npval >= v1) == f'3 >= {s1}'
    assert str(npval != v1) == f'3 != {s1}'
    assert str(npval == v1) == f'3 == {s1}'

    v = f(v1, v2)
    assert str(v) == f'10 + {s1} * 2 + sin({s1} + {s2} * 3)'
    assert repr(v) == str(v)
    assert rtval.get_value(v, 1) == f(1, 2)

    v3 = v1 + 2
    v4 = 1.2 + v2
    assert str(v3) == f'2 + {s1}'
    assert str(v4) == f'1.2 + {s2}'
    assert str(v1 + rtval.new_const(2)) == f'2 + {s1}'
    assert str(rtval.new_const(1.2) + v2) == f'1.2 + {s2}'

    assert str(v3 + v4) == f'3.2 + {s1} + {s2}'
    assert str(v3 - v4) == f'0.8 + {s1} - {s2}'
    assert str(v3 + (1.2 - v2)) == f'3.2 + {s1} - {s2}'
    assert str(v3 - (3 - v2)) == f'-1 + {s1} + {s2}'

    assert str((1.2 - v2) + v3) == f'3.2 + {s1} - {s2}'
    assert str((3 - v2) - v3) == f'1 - ({s2} + {s1})'

    assert str((v1 + v2) * 2) == f'({s1} + {s2}) * 2'
    assert str(2 * (v1 + v2)) == f'2 * ({s1} + {s2})'

values = [True, False, -5, -4, -12, -2, -1, 0, 1, 2, 3, 4, 10,
          0.12, 0.34, 0.56, 1.02, 3.04, -0.12, -0.34, -0.56, -1.02, -3.04,
          np.int32(2), np.float32(3.4)]

def run_check_unary(f):
    for v in values:
        try:
            cv = f(v)
        except:
            continue
        rv = f(rtval.new_extern(lambda: v))
        assert cmp_value(rtval.get_value(rv, 1), cv)

def run_check_binary(f):
    for v1 in values:
        for v2 in values:
            try:
                cv = f(v1, v2)
            except:
                continue
            for rv1 in [v1, rtval.new_extern(lambda: v1)]:
                for rv2 in [v2, rtval.new_extern(lambda: v2)]:
                    rv = f(rv1, rv2)
                    assert cmp_value(rtval.get_value(rv, 1), cv)

def run_check_ternary(f):
    for v1 in values:
        for v2 in values:
            for v3 in values:
                try:
                    cv = f(v1, v2, v3)
                except:
                    continue
                for rv1 in [v1, rtval.new_extern(lambda: v1)]:
                    for rv2 in [v2, rtval.new_extern(lambda: v2)]:
                        for rv3 in [v3, rtval.new_extern(lambda: v3)]:
                            rv = f(rv1, rv2, rv3)
                            assert cmp_value(rtval.get_value(rv, 1), cv)

def sum3(v1, v2, v3):
    return sum([v1, v2, v3])

def test_ops():
    olderr = np.seterr(all='ignore')
    for f in [operator.pos, operator.neg, rtval.inv, np.logical_not, abs, np.abs,
              math.ceil, np.ceil, np.exp, np.expm1, math.floor, np.floor, np.log,
              np.log1p, np.log2, np.log10, np.sqrt, np.arcsin, np.arccos, np.arctan,
              np.arcsinh, np.arccosh, np.arctanh, np.sin, np.cos, np.tan, np.sinh,
              np.cosh, np.tanh, np.rint, rtval.round_int64, rtval.convert_bool]:
        run_check_unary(f)
    # Omit np.add from the list since it behaves differently for booleans
    # Omit np.divide, np.remainder, np.power since they handles 0 differently
    # (and throw off the error detection)
    for f in [operator.add, operator.sub, np.subtract, operator.mul, np.multiply,
              operator.truediv, operator.lt, np.less, operator.gt, np.greater,
              operator.le, np.less_equal, operator.ge, np.greater_equal,
              operator.ne, np.not_equal, operator.eq, np.equal,
              operator.and_, np.bitwise_and, operator.or_, np.bitwise_or,
              operator.xor, np.bitwise_xor, operator.pow, np.arctan2, np.hypot,
              np.fmin, np.fmax, operator.mod]:
        run_check_binary(f)
    for f in [sum3, rtval.ifelse]:
        run_check_ternary(f)
    np.seterr(**olderr)

def test_np_ops():
    assert list(rtval.inv(np.array([True, False]))) == [False, True]
    assert list(rtval.inv(np.array([-2, -1, 0, 1, 2]))) == [False, False, True, False, False]
    assert list(rtval.convert_bool(np.array([True, False]))) == [True, False]
    assert list(rtval.convert_bool(np.array([0, -1, 1]))) == [False, True, True]
    assert list(rtval.convert_bool(np.array([0.0, 1.0, -1.0]))) == [False, True, True]

    a = rtval.round_int64(np.array([-0.9, -0.1, 0.0, -0.1, 0.9]))
    assert a.dtype == np.int64
    assert list(a) == [-1, 0, 0, 0, 1]
    a = rtval.round_int64(np.array([True, False]))
    assert a.dtype == np.int64
    assert list(a) == [1, 0]
    a = rtval.round_int64(np.array([-2, -1, 0, 1, 2], dtype=np.int32))
    assert a.dtype == np.int64
    assert list(a) == [-2, -1, 0, 1, 2]
    a = rtval.round_int64(np.array([-2, -1, 0, 1, 2], dtype=np.int64))
    assert a.dtype == np.int64
    assert list(a) == [-2, -1, 0, 1, 2]

    assert list(rtval.ifelse(True, np.array([1, 2]),
                             np.array([0.1, 0.2]))) == [1.0, 2.0]
    assert list(rtval.ifelse(False, np.array([1, 2]),
                             np.array([0.1, 0.2]))) == [0.1, 0.2]
    assert list(rtval.ifelse(np.array([True, False]), np.array([1, 2]),
                             np.array([0.1, 0.2]))) == [1.0, 0.2]

    assert list(rtval.ifelse(True, 1, np.array([0.1, 0.2]))) == [1.0, 1.0]
    assert list(rtval.ifelse(False, 1, np.array([0.1, 0.2]))) == [0.1, 0.2]
    assert list(rtval.ifelse(np.array([True, False]), 2,
                             np.array([0.1, 0.2]))) == [2.0, 0.2]

    assert list(rtval.ifelse(True, np.array([1, 2]), 0.2)) == [1.0, 2.0]
    assert list(rtval.ifelse(False, np.array([1, 2]), 0.2)) == [0.2, 0.2]
    assert list(rtval.ifelse(np.array([True, False]), np.array([1, 2]),
                             0.2)) == [1.0, 0.2]
    assert list(rtval.ifelse(np.array([True, False]), 2, 2.3)) == [2.0, 2.3]

class C:
    p1 = rtval.RTProp()
    p2 = rtval.RTProp()


def test_rtprop():
    assert isinstance(C.p1, rtval.RTProp)
    assert isinstance(C.p2, rtval.RTProp)

    c1 = C()
    c2 = C()

    prop2 = rtval.RTProp()
    with pytest.raises(ValueError, match="Cannot determine runtime property name"):
        prop2.__get__(c1, C)

    c1p1 = c1.p1
    assert isinstance(c1p1, rtval.RuntimeValue)
    assert str(c1p1).startswith('<RTProp p1 for ')
    assert str(c1p1) == str(c1.p1)

    c2p1 = c2.p1
    assert isinstance(c2p1, rtval.RuntimeValue)
    assert str(c2p1).startswith('<RTProp p1 for ')
    assert str(c2p1) == str(c2.p1)

    with pytest.raises(ValueError):
        c1p1.eval(1)

    with pytest.raises(ValueError):
        c2p1.eval(1)

    c1.p1 = 1.2
    assert c1.p1 == 1.2
    assert c1p1.eval(1) == 1.2
    assert rtval.get_value(c1p1, 1) == 1.2
    assert isinstance(c2.p1, rtval.RuntimeValue)
    assert str(c2p1) == str(c2.p1)

    assert isinstance(C.p1, rtval.RTProp)
    assert isinstance(C.p2, rtval.RTProp)

    v3 = rtval.new_extern(lambda: 3.5)
    c2.p1 = v3
    assert c2.p1 is v3
    assert c2p1.eval(1) == 3.5
    assert rtval.get_value(c2p1, 1) == 3.5

    assert isinstance(c1.p2, rtval.RuntimeValue)
    assert isinstance(c2.p2, rtval.RuntimeValue)
    assert str(c1.p2).startswith('<RTProp p2 for ')
    assert str(c2.p2).startswith('<RTProp p2 for ')

    c1 = C()

    v4 = rtval.new_extern(lambda: 1.25)
    c1.p1 = 2.3
    c1.p2 = v4
    assert c1.p1 == 2.3
    assert c1.p2 is v4

def test_invalid():
    r = test_utils.new_invalid_rtval()
    assert str(r) == "Unknown value"
    with pytest.raises(ValueError, match="Unknown value type"):
        r.eval(1)
