#

from brassboard_seq import rtval
import py_test_utils as test_utils

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
            return v1 == pytest.approx(v2, rel=1e-4, abs=1e-4)
        if isinstance(v2, np.float16):
            return v1 == pytest.approx(v2, rel=1e-3, abs=1e-3)
        # Now the following code doesn't make sense
        # but approx can actually be more restricted than == in some cases...
        # Ref https://github.com/pytest-dev/pytest/pull/9354
        # Ref https://github.com/pytest-dev/pytest/issues/13218
        return v1 == v2 or v1 == pytest.approx(v2)
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

    v1 = test_utils.new_extern(lambda: 1)
    s1 = str(v1)
    assert s1.startswith('extern(')
    v2 = test_utils.new_extern_age(lambda age: 2)
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

    c1 = test_utils.new_const(1.2)
    c2 = test_utils.new_const(2.3)
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
    assert rtval.same_value(1.2, test_utils.new_const(1.2))
    assert rtval.same_value(test_utils.new_const(True), 1.0)
    assert rtval.same_value(test_utils.new_const(10), test_utils.new_const(10.0))
    assert not rtval.same_value(1.2, test_utils.new_const(0.2))
    assert not rtval.same_value(test_utils.new_const(False), 1.0)
    assert not rtval.same_value(test_utils.new_const(10), test_utils.new_const(10.5))

    assert rtval.same_value(test_utils.new_arg(0), test_utils.new_arg(0))
    assert not rtval.same_value(test_utils.new_arg(0), test_utils.new_arg(1))
    assert not rtval.same_value(test_utils.new_extern(lambda: 1), test_utils.new_extern(lambda: 1))
    assert not rtval.same_value(test_utils.new_extern_age(lambda age: 1),
                                test_utils.new_extern_age(lambda age: 1))

    assert rtval.same_value(v1 + v2, v2 + v1)
    assert rtval.same_value(v1 + v2, v2 + v1)
    assert rtval.same_value(-v1, -v1)

    for uop in [rtval.inv, rtval.convert_bool, abs, math.ceil, math.floor,
                np.exp, np.expm1, np.log, np.log1p, np.log2, np.log10, np.sqrt,
                np.arcsin, np.arccos, np.arctan, np.sin, np.cos, np.tan,
                np.sinh, np.cosh, np.tanh, np.rint, round]:
        assert rtval.same_value(uop(v1), uop(v1))
        assert rtval.same_value(uop(v2), uop(v2))
        assert not rtval.same_value(uop(v1), uop(v2))

    for bop in [operator.sub, operator.truediv, operator.pow, operator.mod,
                operator.lt, operator.gt, operator.le, operator.ge, np.arctan2]:
        assert rtval.same_value(bop(v1, v1), bop(v1, v1))
        assert rtval.same_value(bop(v2, v2), bop(v2, v2))
        assert rtval.same_value(bop(v1, v2), bop(v1, v2))
        assert not rtval.same_value(bop(v1, v2), bop(v2, v1))

    for bop in [operator.add, operator.mul, operator.and_, operator.or_, operator.xor,
                operator.ne, operator.eq, np.hypot, np.fmin, np.fmax]:
        assert rtval.same_value(bop(v1, v1), bop(v1, v1))
        assert rtval.same_value(bop(v2, v2), bop(v2, v2))
        assert rtval.same_value(bop(v1, v2), bop(v1, v2))
        assert rtval.same_value(bop(v1, v2), bop(v2, v1))
        if bop not in (operator.ne, operator.eq):
            assert not rtval.same_value(bop(v1, v1), bop(v2, v2))
        assert not rtval.same_value(bop(v1, v2), bop(v1, v1))
        assert not rtval.same_value(bop(v1, v2), bop(v1, 3))

    assert rtval.same_value(rtval.ifelse(v1, v2, 1), rtval.ifelse(v1, v2, 1.0))
    assert not rtval.same_value(rtval.ifelse(v1, v2, 1), rtval.ifelse(v2, v2, 1.0))
    assert not rtval.same_value(rtval.ifelse(v1, v2, 1), rtval.ifelse(v1, v2, 2.0))
    assert not rtval.same_value(rtval.ifelse(v1, v2, 1), rtval.ifelse(v2, v2, 1.0))

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

    r64v1 = round(v1)
    assert str(r64v1) == f'int64({s1})'
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
    assert str(v1 + test_utils.new_const(2)) == f'2 + {s1}'
    assert str(test_utils.new_const(1.2) + v2) == f'1.2 + {s2}'

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

def get_value_options(v):
    if isinstance(v, bool):
        return [v, test_utils.new_extern(lambda: v, bool), test_utils.new_extern(lambda: v)]
    if isinstance(v, int) or isinstance(v, np.integer):
        return [v, test_utils.new_extern(lambda: v, int), test_utils.new_extern(lambda: v)]
    return [v, test_utils.new_extern(lambda: v)]

def run_check_unary(f):
    for v in values:
        try:
            cv = f(v)
        except:
            continue
        for rv1 in get_value_options(v):
            rv = f(rv1)
            try:
                rv = rtval.get_value(rv, 1)
            except:
                assert not np.isfinite(cv) or not np.isreal(cv)
                continue
            assert cmp_value(rv, cv)

def run_check_binary(f):
    for v1 in values:
        for v2 in values:
            try:
                cv = f(v1, v2)
            except:
                continue
            for rv1 in get_value_options(v1):
                for rv2 in get_value_options(v2):
                    rv = f(rv1, rv2)
                    try:
                        rv = rtval.get_value(rv, 1)
                    except:
                        if v2 == 0 and (f is operator.mod or f is np.remainder):
                            # Numpy doesn't raise an error in this case for some reason
                            continue
                        assert not np.isfinite(cv) or not np.isreal(cv)
                        continue
                    assert cmp_value(rv, cv)

def run_check_ternary(f):
    for v1 in values:
        for v2 in values:
            for v3 in values:
                try:
                    cv = f(v1, v2, v3)
                except:
                    continue
                for rv1 in get_value_options(v1):
                    for rv2 in get_value_options(v2):
                        for rv3 in get_value_options(v3):
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
              np.cosh, np.tanh, np.rint, round, rtval.convert_bool]:
        run_check_unary(f)
    # Omit np.add from the list since it behaves differently for booleans
    for f in [operator.add, operator.sub, np.subtract, operator.mul, np.multiply,
              operator.truediv, np.divide, operator.lt, np.less, operator.gt,
              np.greater, operator.le, np.less_equal, operator.ge, np.greater_equal,
              operator.ne, np.not_equal, operator.eq, np.equal,
              operator.and_, np.bitwise_and, operator.or_, np.bitwise_or,
              operator.xor, np.bitwise_xor, operator.pow, np.power, np.arctan2,
              np.hypot, np.fmin, np.fmax, operator.mod, np.remainder]:
        run_check_binary(f)
    for f in [sum3, rtval.ifelse]:
        run_check_ternary(f)
    np.seterr(**olderr)

class C:
    p1 = rtval.RTProp()
    p2 = rtval.RTProp()

    cp_invalid = rtval.CompositeRTProp("not callable")

    cp_scalar_count = 0
    @rtval.CompositeRTProp
    def cp_scalar(self):
        C.cp_scalar_count += 1
        return 10

    cp_list_count = 0
    @rtval.CompositeRTProp
    def cp_list(self):
        C.cp_list_count += 1
        return [0, 1, 2, 3]

    cp_dict_count = 0
    @rtval.CompositeRTProp
    def cp_dict(self):
        C.cp_dict_count += 1
        return {'a': 1, 'b': 2, 3: 10}

    cp_nest_count = 0
    @rtval.CompositeRTProp
    def cp_nest(self):
        C.cp_nest_count += 1
        return {'a': 1, 'b': 2, 3: 10, -4: [1, 2, 3, {'b': [2, 3]}]}


def test_rtprop():
    assert isinstance(C.p1, rtval.RTProp)
    assert isinstance(C.p2, rtval.RTProp)

    with pytest.raises(TypeError):
        rtval.RTProp(1)
    with pytest.raises(TypeError):
        rtval.RTProp(a=2)
    with pytest.raises(TypeError):
        rtval.CompositeRTProp()
    with pytest.raises(TypeError):
        rtval.CompositeRTProp(a=2)

    c1 = C()
    c2 = C()

    prop2 = rtval.RTProp()
    with pytest.raises(ValueError, match="Cannot determine runtime property name"):
        prop2.__get__(c1, C)
    with pytest.raises(ValueError, match="Cannot determine runtime property name"):
        prop2.__set__(c1, 1)

    assert C.p1.get_state(c1) is None
    c1p1 = c1.p1
    assert isinstance(c1p1, rtval.RuntimeValue)
    assert str(c1p1).startswith('<RTProp p1 for ')
    assert str(c1p1) == str(c1.p1)
    assert C.p1.get_state(c1) is c1p1

    assert C.p1.get_state(c2) is None
    c2p1 = c2.p1
    assert isinstance(c2p1, rtval.RuntimeValue)
    assert str(c2p1).startswith('<RTProp p1 for ')
    assert str(c2p1) == str(c2.p1)
    assert C.p1.get_state(c2) is c2p1

    with pytest.raises(ValueError):
        c1p1.eval(1)

    with pytest.raises(ValueError):
        c2p1.eval(1)

    c1.p1 = 1.2
    assert c1.p1 == 1.2
    assert C.p1.get_state(c1) == 1.2
    assert C.p1.get_state(c2) is c2p1
    assert c1p1.eval(1) == 1.2
    assert rtval.get_value(c1p1, 1) == 1.2
    assert isinstance(c2.p1, rtval.RuntimeValue)
    assert str(c2p1) == str(c2.p1)

    assert isinstance(C.p1, rtval.RTProp)
    assert isinstance(C.p2, rtval.RTProp)

    v3 = test_utils.new_extern(lambda: 3.5)
    c2.p1 = v3
    assert c2.p1 is v3
    assert C.p1.get_state(c1) == 1.2
    assert C.p1.get_state(c2) is v3
    assert c2p1.eval(1) == 3.5
    assert rtval.get_value(c2p1, 1) == 3.5

    assert isinstance(c1.p2, rtval.RuntimeValue)
    assert isinstance(c2.p2, rtval.RuntimeValue)
    assert str(c1.p2).startswith('<RTProp p2 for ')
    assert str(c2.p2).startswith('<RTProp p2 for ')

    C.p1.set_state(c1, None)
    assert C.p1.get_state(c1) is None
    c1p1_2 = c1.p1
    assert C.p1.get_state(c1) is c1p1_2
    assert c1p1 is not c1p1_2
    assert c1.p1 is c1p1_2
    C.p1.set_state(c1, 3)
    assert rtval.get_value(c1p1, 3) == 3
    assert rtval.get_value(c1p1_2, 3) == 3

    c1 = C()

    v4 = test_utils.new_extern(lambda: 1.25)
    c1.p1 = 2.3
    c1.p2 = v4
    assert c1.p1 == 2.3
    assert c1.p2 is v4

    del c1.p1
    del c1.p2
    assert isinstance(c1.p1, rtval.RuntimeValue)
    assert isinstance(c1.p2, rtval.RuntimeValue)

def test_composite_rtprop():
    C.cp_scalar_count = 0
    C.cp_list_count = 0
    C.cp_dict_count = 0
    C.cp_nest_count = 0
    assert isinstance(C.cp_invalid, rtval.CompositeRTProp)
    assert isinstance(C.cp_scalar, rtval.CompositeRTProp)
    assert isinstance(C.cp_list, rtval.CompositeRTProp)
    assert isinstance(C.cp_dict, rtval.CompositeRTProp)
    assert isinstance(C.cp_nest, rtval.CompositeRTProp)
    assert C.cp_scalar_count == 0
    assert C.cp_list_count == 0
    assert C.cp_dict_count == 0
    assert C.cp_nest_count == 0

    c1 = C()
    c2 = C()

    cprop2 = rtval.CompositeRTProp(lambda obj: 1)
    with pytest.raises(ValueError, match="Cannot determine runtime property name"):
        cprop2.__get__(c1, C)

    with pytest.raises(TypeError):
        c1.cp_invalid
    assert c1.cp_scalar == 10
    assert C.cp_scalar_count == 1
    assert c1.cp_scalar == 10
    assert C.cp_scalar_count == 1
    assert c2.cp_scalar == 10
    assert C.cp_scalar_count == 2
    assert c2.cp_scalar == 10
    assert C.cp_scalar_count == 2

    setattr(c1, '__CompositeRTProp__cp_scalar', 10)
    assert c1.cp_scalar == 10
    assert C.cp_scalar_count == 3
    delattr(c1, '__CompositeRTProp__cp_scalar')
    assert c1.cp_scalar == 10
    assert C.cp_scalar_count == 4

    assert C.cp_scalar.get_state(c1) is None

    C.cp_scalar.set_state(c1, [3, 4])
    assert C.cp_scalar.get_state(c1) == [3, 4]
    assert c1.cp_scalar == [3, 4]
    assert c2.cp_scalar == 10
    assert C.cp_scalar_count == 4

    C.cp_scalar.set_state(c1, {1: 1000})
    assert C.cp_scalar.get_state(c1) == {1: 1000}
    with pytest.raises(TypeError, match=f"Unknown value type '{int}'"):
        c1.cp_scalar
    assert c2.cp_scalar == 10
    assert C.cp_scalar_count == 4

    C.cp_scalar.set_state(c1, {})
    assert C.cp_scalar.get_state(c1) == {}
    assert c1.cp_scalar == 10
    assert c2.cp_scalar == 10
    assert C.cp_scalar_count == 4

    C.cp_scalar.set_state(c1, None)
    assert C.cp_scalar.get_state(c1) is None
    assert c1.cp_scalar == 10
    assert c2.cp_scalar == 10
    assert C.cp_scalar_count == 4

    c1_cp_list1 = c1.cp_list
    assert c1_cp_list1 == [0, 1, 2, 3]
    assert C.cp_list_count == 1
    assert c1.cp_list is c1_cp_list1
    assert C.cp_list_count == 1
    c2_cp_list1 = c2.cp_list
    assert c2_cp_list1 == [0, 1, 2, 3]
    assert C.cp_list_count == 2
    assert c2.cp_list is c2_cp_list1
    assert C.cp_list_count == 2

    assert C.cp_list.get_state(c1) is None

    C.cp_list.set_state(c1, [3, 4])
    assert C.cp_list.get_state(c1) == [3, 4]
    assert c1.cp_list == [3, 4]
    assert c2.cp_list == [0, 1, 2, 3]
    assert C.cp_list_count == 2

    C.cp_list.set_state(c1, {1: 1000})
    assert C.cp_list.get_state(c1) == {1: 1000}
    assert c1.cp_list == [0, 1000, 2, 3]
    assert c2.cp_list == [0, 1, 2, 3]
    assert C.cp_list_count == 2

    C.cp_list.set_state(c1, {'2': 1000})
    assert C.cp_list.get_state(c1) == {'2': 1000}
    assert c1.cp_list == [0, 1, 1000, 3]
    assert c2.cp_list == [0, 1, 2, 3]
    assert C.cp_list_count == 2

    C.cp_list.set_state(c1, {'a': 1000})
    assert C.cp_list.get_state(c1) == {'a': 1000}
    with pytest.raises(ValueError):
        c1.cp_list
    C.cp_list.set_state(c1, {'-1': 1000})
    with pytest.raises(IndexError):
        c1.cp_list
    C.cp_list.set_state(c1, {-1: 1000})
    with pytest.raises(IndexError):
        c1.cp_list
    C.cp_list.set_state(c1, {'10': 1000})
    with pytest.raises(IndexError):
        c1.cp_list
    C.cp_list.set_state(c1, {10: 1000})
    with pytest.raises(IndexError):
        c1.cp_list
    C.cp_list.set_state(c1, {})
    assert C.cp_list.get_state(c1) == {}
    assert c1.cp_list == [0, 1, 2, 3]
    assert c2.cp_list == [0, 1, 2, 3]
    assert C.cp_list_count == 2

    C.cp_list.set_state(c1, None)
    assert C.cp_list.get_state(c1) is None
    assert c1.cp_list == [0, 1, 2, 3]
    assert c2.cp_list == [0, 1, 2, 3]
    assert C.cp_list_count == 2

    c1_cp_dict1 = c1.cp_dict
    assert c1_cp_dict1 == {'a': 1, 'b': 2, 3: 10}
    assert C.cp_dict_count == 1
    assert c1.cp_dict is c1_cp_dict1
    assert C.cp_dict_count == 1
    c2_cp_dict1 = c2.cp_dict
    assert c2_cp_dict1 == {'a': 1, 'b': 2, 3: 10}
    assert C.cp_dict_count == 2
    assert c2.cp_dict is c2_cp_dict1
    assert C.cp_dict_count == 2

    assert C.cp_dict.get_state(c1) is None

    C.cp_dict.set_state(c1, [3, 4])
    assert C.cp_dict.get_state(c1) == [3, 4]
    assert c1.cp_dict == [3, 4]
    assert c2.cp_dict == {'a': 1, 'b': 2, 3: 10}
    assert C.cp_dict_count == 2

    C.cp_dict.set_state(c1, {1: 1000})
    assert C.cp_dict.get_state(c1) == {1: 1000}
    assert c1.cp_dict == {'a': 1, 'b': 2, 1: 1000, 3: 10}
    assert c2.cp_dict == {'a': 1, 'b': 2, 3: 10}
    assert C.cp_dict_count == 2

    C.cp_dict.set_state(c1, {'3': 1000})
    assert C.cp_dict.get_state(c1) == {'3': 1000}
    assert c1.cp_dict == {'a': 1, 'b': 2, 3: 1000}
    assert c2.cp_dict == {'a': 1, 'b': 2, 3: 10}
    assert C.cp_dict_count == 2

    C.cp_dict.set_state(c1, {'a': 1000})
    assert C.cp_dict.get_state(c1) == {'a': 1000}
    assert c1.cp_dict == {'a': 1000, 'b': 2, 3: 10}
    assert c2.cp_dict == {'a': 1, 'b': 2, 3: 10}
    assert C.cp_dict_count == 2

    C.cp_dict.set_state(c1, {'k': 2000})
    assert C.cp_dict.get_state(c1) == {'k': 2000}
    assert c1.cp_dict == {'a': 1, 'b': 2, 3: 10, 'k': 2000}
    assert c2.cp_dict == {'a': 1, 'b': 2, 3: 10}
    assert C.cp_dict_count == 2

    C.cp_dict.set_state(c1, {})
    assert C.cp_dict.get_state(c1) == {}
    assert c1.cp_dict == {'a': 1, 'b': 2, 3: 10}
    assert c2.cp_dict == {'a': 1, 'b': 2, 3: 10}
    assert C.cp_dict_count == 2

    C.cp_dict.set_state(c1, None)
    assert C.cp_dict.get_state(c1) is None
    assert c1.cp_dict == {'a': 1, 'b': 2, 3: 10}
    assert c2.cp_dict == {'a': 1, 'b': 2, 3: 10}
    assert C.cp_dict_count == 2

    c1_cp_nest1 = c1.cp_nest
    assert c1_cp_nest1 == {'a': 1, 'b': 2, 3: 10, -4: [1, 2, 3, {'b': [2, 3]}]}
    assert C.cp_nest_count == 1
    assert c1.cp_nest is c1_cp_nest1
    assert C.cp_nest_count == 1
    c2_cp_nest1 = c2.cp_nest
    assert c2_cp_nest1 == {'a': 1, 'b': 2, 3: 10, -4: [1, 2, 3, {'b': [2, 3]}]}
    assert C.cp_nest_count == 2
    assert c2.cp_nest is c2_cp_nest1
    assert C.cp_nest_count == 2

    assert C.cp_nest.get_state(c1) is None

    C.cp_nest.set_state(c1, [3, 4])
    assert C.cp_nest.get_state(c1) == [3, 4]
    assert c1.cp_nest == [3, 4]
    assert c2.cp_nest == {'a': 1, 'b': 2, 3: 10, -4: [1, 2, 3, {'b': [2, 3]}]}
    assert C.cp_nest_count == 2

    C.cp_nest.set_state(c1, {1: 1000})
    assert C.cp_nest.get_state(c1) == {1: 1000}
    assert c1.cp_nest == {'a': 1, 'b': 2, 1: 1000, 3: 10, -4: [1, 2, 3, {'b': [2, 3]}]}
    assert c2.cp_nest == {'a': 1, 'b': 2, 3: 10, -4: [1, 2, 3, {'b': [2, 3]}]}
    assert C.cp_nest_count == 2

    C.cp_nest.set_state(c1, {'3': 1000})
    assert C.cp_nest.get_state(c1) == {'3': 1000}
    assert c1.cp_nest == {'a': 1, 'b': 2, 3: 1000, -4: [1, 2, 3, {'b': [2, 3]}]}
    assert c2.cp_nest == {'a': 1, 'b': 2, 3: 10, -4: [1, 2, 3, {'b': [2, 3]}]}
    assert C.cp_nest_count == 2

    C.cp_nest.set_state(c1, {'a': 1000})
    assert C.cp_nest.get_state(c1) == {'a': 1000}
    assert c1.cp_nest == {'a': 1000, 'b': 2, 3: 10, -4: [1, 2, 3, {'b': [2, 3]}]}
    assert c2.cp_nest == {'a': 1, 'b': 2, 3: 10, -4: [1, 2, 3, {'b': [2, 3]}]}
    assert C.cp_nest_count == 2

    C.cp_nest.set_state(c1, {'-4': 1000})
    assert C.cp_nest.get_state(c1) == {'-4': 1000}
    assert c1.cp_nest == {'a': 1, 'b': 2, 3: 10, -4: 1000}
    assert c2.cp_nest == {'a': 1, 'b': 2, 3: 10, -4: [1, 2, 3, {'b': [2, 3]}]}
    assert C.cp_nest_count == 2

    C.cp_nest.set_state(c1, {'-4': {'2': 'aaa'}})
    assert C.cp_nest.get_state(c1) == {'-4': {'2': 'aaa'}}
    assert c1.cp_nest == {'a': 1, 'b': 2, 3: 10, -4: [1, 2, 'aaa', {'b': [2, 3]}]}
    assert c2.cp_nest == {'a': 1, 'b': 2, 3: 10, -4: [1, 2, 3, {'b': [2, 3]}]}
    assert C.cp_nest_count == 2

    C.cp_nest.set_state(c1, {'-4': {'3': {'b': {'0': [0]}}}})
    assert C.cp_nest.get_state(c1) == {'-4': {'3': {'b': {'0': [0]}}}}
    assert c1.cp_nest == {'a': 1, 'b': 2, 3: 10, -4: [1, 2, 3, {'b': [[0], 3]}]}
    assert c2.cp_nest == {'a': 1, 'b': 2, 3: 10, -4: [1, 2, 3, {'b': [2, 3]}]}
    assert C.cp_nest_count == 2

    C.cp_nest.set_state(c1, {})
    assert C.cp_nest.get_state(c1) == {}
    assert c1.cp_nest == {'a': 1, 'b': 2, 3: 10, -4: [1, 2, 3, {'b': [2, 3]}]}
    assert c2.cp_nest == {'a': 1, 'b': 2, 3: 10, -4: [1, 2, 3, {'b': [2, 3]}]}
    assert C.cp_nest_count == 2

    C.cp_nest.set_state(c1, None)
    assert C.cp_nest.get_state(c1) is None
    assert c1.cp_nest == {'a': 1, 'b': 2, 3: 10, -4: [1, 2, 3, {'b': [2, 3]}]}
    assert c2.cp_nest == {'a': 1, 'b': 2, 3: 10, -4: [1, 2, 3, {'b': [2, 3]}]}
    assert C.cp_nest_count == 2

    c1._bb_rt_values = {}
    assert c1.cp_scalar == 10
    assert c2.cp_scalar == 10
    assert C.cp_scalar_count == 4

    assert c1.cp_list is c1_cp_list1
    assert c2.cp_list is c2_cp_list1
    assert C.cp_list_count == 2

    assert c1.cp_dict is c1_cp_dict1
    assert c2.cp_dict is c2_cp_dict1
    assert C.cp_dict_count == 2

    assert c1.cp_nest is c1_cp_nest1
    assert c2.cp_nest is c2_cp_nest1
    assert C.cp_nest_count == 2

    c1._bb_rt_values = None
    assert c1.cp_scalar == 10
    assert C.cp_scalar_count == 5
    assert c2.cp_scalar == 10
    assert C.cp_scalar_count == 5

    assert c1.cp_list is not c1_cp_list1
    assert c1.cp_list == c1_cp_list1
    assert C.cp_list_count == 3
    assert c2.cp_list is c2_cp_list1
    assert C.cp_list_count == 3

    assert c1.cp_dict is not c1_cp_dict1
    assert c1.cp_dict == c1_cp_dict1
    assert C.cp_dict_count == 3
    assert c2.cp_dict is c2_cp_dict1
    assert C.cp_dict_count == 3

    assert c1.cp_nest is not c1_cp_nest1
    assert c1.cp_nest == c1_cp_nest1
    assert C.cp_nest_count == 3
    assert c2.cp_nest is c2_cp_nest1
    assert C.cp_nest_count == 3

    # Create a cycle involving composite rtprop data
    C.cp_nest.set_state(c1, c1)

def test_invalid():
    r = test_utils.new_invalid_rtval()
    assert str(r) == "Unknown value"
    with pytest.raises(ValueError, match="Unknown value type"):
        r.eval(1)

def test_logical():
    v1 = 1.0
    v2 = 2
    r1 = test_utils.new_extern(lambda: v1)
    r2 = test_utils.new_extern(lambda: v2)

    r_or = r1 | r2
    r_and = r1 & r2
    r_xor = r1 ^ r2
    assert r_or.eval(0) == 3
    assert isinstance(r_or.eval(0), int)
    assert r_and.eval(0) == 0
    assert isinstance(r_and.eval(0), int)
    assert r_xor.eval(0) == 3
    assert isinstance(r_xor.eval(0), int)

    v1 = 1.2
    with pytest.raises(ValueError, match="bitwise operation on floating point numbers"):
        r_or.eval(1)
    with pytest.raises(ValueError, match="bitwise operation on floating point numbers"):
        r_and.eval(1)
    with pytest.raises(ValueError, match="bitwise operation on floating point numbers"):
        r_xor.eval(1)

    v1 = 1
    v2 = 1.2
    with pytest.raises(ValueError, match="bitwise operation on floating point numbers"):
        r_or.eval(2)
    with pytest.raises(ValueError, match="bitwise operation on floating point numbers"):
        r_and.eval(2)
    with pytest.raises(ValueError, match="bitwise operation on floating point numbers"):
        r_xor.eval(2)

    v1 = 0b1100
    v2 = 0b1010
    assert r_or.eval(3) == 0b1110
    assert isinstance(r_or.eval(3), int)
    assert r_and.eval(3) == 0b1000
    assert isinstance(r_and.eval(3), int)
    assert r_xor.eval(3) == 0b0110
    assert isinstance(r_xor.eval(3), int)

def test_pow():
    v1 = -1
    v2 = 0.5
    r1 = test_utils.new_extern(lambda: v1)
    r2 = test_utils.new_extern(lambda: v2)

    with pytest.raises(TypeError):
        pow(r2, 2, 2)

    rpow = r1**r2
    with pytest.raises(ValueError, match="power of negative number"):
        rpow.eval(0)

    v1 = 1
    assert rpow.eval(1) == 1.0

    v1 = 2.0
    v2 = math.nan
    assert math.isnan(rpow.eval(2))

    v1 = 0
    v2 = -1
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        rpow.eval(3)

def test_round():
    for v in [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        vi = round(v)
        rv = test_utils.new_extern(lambda: v)
        rvi = round(rv)
        assert rvi.eval(0) == vi

def test_error_propagate():
    v1 = 1
    v2 = 0
    r1 = test_utils.new_extern(lambda: v1)
    r2 = test_utils.new_extern(lambda: v2)
    op1 = r1 / r2

    v3 = 2
    r3 = test_utils.new_extern(lambda: v3)
    op2 = np.arcsin(r3)

    b = True
    rb = test_utils.new_extern(lambda: b, bool)

    res = op1 + op2
    sel = rtval.ifelse(rb, op1, op2)
    res2 = abs(op2)

    with pytest.raises(ZeroDivisionError, match="division by zero"):
        res.eval(0)
    b = True
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        sel.eval(0)
    b = False
    with pytest.raises(ValueError, match="math domain error"):
        sel.eval(1)
    with pytest.raises(ValueError, match="math domain error"):
        res2.eval(1)

    v3 = 0.1
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        res.eval(2)
    b = True
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        sel.eval(2)
    b = False
    assert sel.eval(3) == pytest.approx(np.arcsin(v3))
    assert res2.eval(3) == pytest.approx(np.arcsin(v3))

    v2 = 0.2
    assert res.eval(4) == pytest.approx(v1 / v2 + np.arcsin(v3))
    b = True
    assert sel.eval(4) == pytest.approx(v1 / v2)
    b = False
    assert sel.eval(5) == pytest.approx(np.arcsin(v3))
    assert res2.eval(5) == pytest.approx(np.arcsin(v3))

    v3 = 3
    with pytest.raises(ValueError, match="math domain error"):
        res.eval(6)
    b = True
    assert sel.eval(6) == pytest.approx(v1 / v2)
    b = False
    with pytest.raises(ValueError, match="math domain error"):
        sel.eval(7)
    with pytest.raises(ValueError, match="math domain error"):
        res2.eval(7)

def test_type():
    for v in [True, 1, 1.0]:
        vb = test_utils.new_extern(lambda: v, bool)
        vi = test_utils.new_extern(lambda: v, int)
        vf = test_utils.new_extern(lambda: v, float)
        assert vb.eval(0) is True
        assert isinstance(vb.eval(0), bool)
        assert vi.eval(0) == 1
        assert isinstance(vi.eval(0), int)
        assert vf.eval(0) == 1.0
        assert isinstance(vf.eval(0), float)
        vb = test_utils.new_extern_age(lambda age: v, bool)
        vi = test_utils.new_extern_age(lambda age: v, int)
        vf = test_utils.new_extern_age(lambda age: v, float)
        assert vb.eval(0) is True
        assert isinstance(vb.eval(0), bool)
        assert vi.eval(0) == 1
        assert isinstance(vi.eval(0), int)
        assert vf.eval(0) == 1.0
        assert isinstance(vf.eval(0), float)

    with pytest.raises(TypeError, match=f"Unknown runtime value type '{list}'"):
        test_utils.new_extern(lambda: True, list)

def test_arg():
    r = test_utils.new_arg(0)
    assert str(r) == "arg(0)"
    with pytest.raises(ValueError, match="Cannot evaluate unknown argument"):
        r.eval(1)
