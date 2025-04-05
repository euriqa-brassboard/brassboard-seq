#

from brassboard_seq import action, rtval
import py_test_utils as test_utils
import pytest

import math
import numpy as np
import operator

def check_action_str(action, s):
    assert str(action) == s
    assert repr(action) == s

def test_action():
    a1 = test_utils.new_action(1.2, True, True, False, {}, 0)
    test_utils.action_set_tid(a1, 1)
    assert test_utils.action_get_aid(a1) == 0
    a2 = test_utils.new_action(1.2, True, False, False, dict(a=1, b=2), 3)
    assert test_utils.action_get_aid(a2) == 3
    test_utils.action_set_tid(a2, 3)
    a3 = test_utils.new_action(2.3, False, True, True, {}, 2)
    assert test_utils.action_get_aid(a3) == 2
    test_utils.action_set_tid(a3, 2)
    check_action_str(a1, "Pulse(1.2)")
    check_action_str(a2, "Set(1.2, a=1, b=2)")
    check_action_str(a3, "Pulse(2.3, cond=False, exact_time=True)")

def test_ramp_base():
    r = action._RampFunctionBase()
    test = test_utils.RampTest(r, 1, 0)
    with pytest.raises(NotImplementedError):
        test.eval_compile_end()
    with pytest.raises(NotImplementedError):
        test.eval_runtime(1, [1, 2, 3])
    with pytest.raises(NotImplementedError):
        test_utils.ramp_get_spline_segments(r, 1, 0)
    assert test_utils.ramp_runtime_eval(r, 1) == 0

def test_ramp_eval():
    rt = test_utils.new_extern(lambda: 1)
    rlen = test_utils.new_extern(lambda: 2)
    rold = test_utils.new_extern(lambda: -0.2)
    st = str(rt)
    slen = str(rlen)
    sold = str(rold)

    test = test_utils.RampTest(test_utils.StaticFunction(), rlen, rold)
    rv = test.eval_compile_end()
    assert str(rv) == f'({slen} / 2 + {sold}) - {slen}'
    assert rtval.get_value(rv, 0) == 2 / 2 + -0.2 - 2

    ts = np.linspace(-1, 1, 1000)
    vs = test.eval_runtime(0, ts)
    assert (vs == ts / 2 + -0.2 - 2).all()

    ts = np.linspace(-1, 1, 100)
    vs = test.eval_runtime(0, ts)
    assert (vs == ts / 2 + -0.2 - 2).all()

    ts = np.linspace(-1, 1, 2000)
    vs = test.eval_runtime(0, ts)
    assert (vs == ts / 2 + -0.2 - 2).all()

    ramp = test_utils.new_extern(lambda: 2)
    rfreq = test_utils.new_extern(lambda: 1.2)
    rphase = 2.3
    samp = str(ramp)
    sfreq = str(rfreq)
    test = test_utils.RampTest(test_utils.SinFunction(ramp, rfreq, rphase), rlen, rold)
    rv = test.eval_compile_end()
    assert str(rv) == f'{sold} + {samp} * sin(2.3 + {sfreq} * {slen})'
    assert rtval.get_value(rv, 0) == -0.2 + 2 * np.sin(2.3 + 1.2 * 2)

    ts = np.linspace(-1, 1, 1000)
    vs = test.eval_runtime(0, ts)
    assert (vs == -0.2 + 2 * np.sin(2.3 + 1.2 * ts)).all()

    ts = np.linspace(-1, 1, 100)
    vs = test.eval_runtime(0, ts)
    assert (vs == -0.2 + 2 * np.sin(2.3 + 1.2 * ts)).all()

    ts = np.linspace(-1, 1, 2000)
    vs = test.eval_runtime(0, ts)
    assert (vs == -0.2 + 2 * np.sin(2.3 + 1.2 * ts)).all()

    with pytest.raises(TypeError):
        test_utils.ValueFunction([])
    wtfunc = test_utils.ValueFunction(1)
    test = test_utils.RampTest(wtfunc, rlen, rold)
    assert test.eval_compile_end() == 1
    assert test.eval_runtime(0, [1, 2, 3]) == [1, 1, 1]
    with pytest.raises(TypeError):
        test_utils.ValueFunction(np.empty((2, 2)))
    with pytest.raises(IndexError, match="Argument index out of bound: 3."):
        test_utils.ValueFunction(test_utils.new_arg(3))
    with pytest.raises(TypeError):
        test_utils.ValueFunction(test_utils.new_arg([]))
    with pytest.raises(ValueError, match="Unknown value type"):
        test_utils.ValueFunction(test_utils.new_invalid_rtval())

    with pytest.raises(ValueError, match="^AAAAA$"):
        test_utils.ErrorFunction(ValueError("AAAAA"))

def test_spline():
    assert test_utils.ramp_get_spline_segments(test_utils.SinFunction(1.0, 2.0, 0.1), 1, 0) is None

    o0 = 0.1
    o1 = 0.2
    o2 = 1
    o3 = 0.3

    def check_spline(rt0, rt1, rt2, rt3):
        ro0 = test_utils.new_extern(lambda: o0) if rt0 else o0
        ro1 = test_utils.new_extern(lambda: o1) if rt1 else o1
        ro2 = test_utils.new_extern(lambda: o2) if rt2 else o2
        ro3 = test_utils.new_extern(lambda: o3) if rt3 else o3

        sp_seq = action.SeqCubicSpline(ro0, ro1, ro2, ro3)
        assert sp_seq.order0 is ro0
        assert sp_seq.order1 is ro1
        assert sp_seq.order2 is ro2
        assert sp_seq.order3 is ro3
        sp_py = test_utils.PyCubicSpline(ro0, ro1, ro2, ro3)
        assert test_utils.ramp_get_spline_segments(sp_seq, 1, 0) == ()
        assert test_utils.ramp_get_spline_segments(sp_py, 1, 0) == ()

        test_seq = test_utils.RampTest(sp_seq, 1, 0.2)
        test_py = test_utils.RampTest(sp_py, 1, 0.2)

        v1_seq = test_seq.eval_compile_end()
        assert rtval.get_value(v1_seq, 1) == 1.6

        v1_py = test_py.eval_compile_end()
        assert rtval.get_value(v1_py, 1) == 1.6

        assert list(test_seq.eval_runtime(2, [0, 0.5, 1])) == pytest.approx([0.1, 0.4875, 1.6])
        assert list(test_py.eval_runtime(2, [0, 0.5, 1])) == pytest.approx([0.1, 0.4875, 1.6])

    for i in range(15):
        check_spline(i & 1, i & 2, i & 4, i & 8)

def blackman_func(t): # t in [0, 1]
    theta = (t - 0.5) * 2 * np.pi
    return 21/50 + 1/2 * np.cos(theta) + 2/25 * np.cos(2 * theta)

def test_blackman():
    bp = action.Blackman(0.9, 0.1)
    assert bp.amp == 0.9
    assert bp.offset == 0.1
    test1 = test_utils.RampTest(bp, 1, 0)
    bsp = action.BlackmanSquare(0.6, -0.1)
    assert bsp.amp == 0.6
    assert bsp.offset == -0.1
    test2 = test_utils.RampTest(bsp, 1, 0)
    ts = np.linspace(0, 1, 1000)
    v1 = test1.eval_runtime(0, ts)
    v2 = test2.eval_runtime(0, ts)
    bs = blackman_func(ts)
    expect1 = list(0.1 + 0.9 * bs)
    expect2 = list(-0.1 + 0.6 * bs**2)
    assert v1 == pytest.approx(expect1)
    assert v2 == pytest.approx(expect2)

def test_linear():
    lp = action.LinearRamp(0.2, 0.8)
    assert lp.start == 0.2
    assert lp.end == 0.8
    test1 = test_utils.RampTest(lp, 1, 0)
    ts = np.linspace(0, 1, 1000)
    v1 = test1.eval_runtime(0, ts)
    expect1 = list(ts * 0.6 + 0.2)
    assert v1 == pytest.approx(expect1)

def test_const():
    ts = np.linspace(0, 1, 1000)

    p = test_utils.ValueFunction(0.2)
    test1 = test_utils.RampTest(p, 1, 0)
    v1 = test1.eval_runtime(0, ts)
    expect1 = list(ts * 0 + 0.2)
    assert v1 == pytest.approx(expect1)

    p = test_utils.ValueFunction(1)
    test1 = test_utils.RampTest(p, 1, 0)
    v1 = test1.eval_runtime(0, ts)
    expect1 = list(ts * 0 + 1)
    assert v1 == pytest.approx(expect1)

    v0 = 0.1
    p = test_utils.ValueFunction(test_utils.new_extern(lambda: v0))
    test1 = test_utils.RampTest(p, 1, 0)
    v1 = test1.eval_runtime(0, ts)
    expect1 = list(ts * 0 + v0)
    assert v1 == pytest.approx(expect1)

    v0 = 0.2
    v1 = test1.eval_runtime(1, ts)
    expect1 = list(ts * 0 + v0)
    assert v1 == pytest.approx(expect1)

    v0 = 0.1
    p = test_utils.ValueFunction(np.cos(test_utils.new_extern(lambda: v0)) + 1)
    test1 = test_utils.RampTest(p, 1, 0)
    v1 = test1.eval_runtime(0, ts)
    expect1 = list(ts * 0 + np.cos(v0) + 1)
    assert v1 == pytest.approx(expect1)

    v0 = 0.2
    v1 = test1.eval_runtime(1, ts)
    expect1 = list(ts * 0 + np.cos(v0) + 1)
    assert v1 == pytest.approx(expect1)

values = [True, False, -5, -4, -12, -2, -1, 0, 1, 2, 3, 4, 10,
          0.12, 0.34, 0.56, 1.02, 3.04, -0.12, -0.34, -0.56, -1.02, -3.04]

def throw_non_finite(f, *args):
    res = f(*args)
    if not np.isfinite(res) or not np.isreal(res):
        raise ValueError("domain error")
    return res

def none_on_error(cb, *args):
    try:
        return cb(*args)
    except:
        return None

def cmp_list(l1, l2):
    if l1 is None:
        return l1 is l2
    for v1, v2 in zip(l1, l2):
        assert np.isfinite(v1)
        assert np.isfinite(v2)
        if isinstance(v2, np.float32):
            if not v1 == pytest.approx(v2, rel=1e-4, abs=1e-4):
                return False
        elif isinstance(v2, np.float16):
            if not v1 == pytest.approx(v2, rel=1e-3, abs=1e-3):
                return False
        # Now the following code doesn't make sense
        # but approx can actually be more restricted than == in some cases...
        # Ref https://github.com/pytest-dev/pytest/pull/9354
        # Ref https://github.com/pytest-dev/pytest/issues/13218
        elif not (v1 == v2 or v1 == pytest.approx(v2)):
            return False
    return True

def run_check_unary(f):
    if f is np.arccosh:
        ts1 = np.linspace(1.1, 2.5, 1000)
    else:
        ts1 = np.linspace(0.1, 0.5, 1000)
    ts2 = np.linspace(-2, 2, 1001)

    p1 = test_utils.FuncAction(lambda t, l, o: f(t))
    test1 = test_utils.RampTest(p1, 2, 0)
    v1 = test1.eval_runtime(0, ts1)
    expect1 = [f(t) for t in ts1]
    assert cmp_list(v1, expect1)
    v2 = none_on_error(test1.eval_runtime, 0, ts2)
    @none_on_error
    def expect2():
        return [throw_non_finite(f, t) for t in ts2]
    assert cmp_list(v2, expect2)

    ts = np.linspace(0, 2, 3)
    for v in values:
        rv = test_utils.new_extern(lambda: v)
        @none_on_error
        def expect1():
            fv = throw_non_finite(f, v)
            return [fv for t in ts]
        p2 = test_utils.FuncAction(lambda t, l, o: f(rv))
        test2 = test_utils.RampTest(p2, 2, 0)
        v1 = none_on_error(test2.eval_runtime, 0, ts)
        assert cmp_list(v1, expect1)

def run_check_binary(f):
    ts1 = np.linspace(0, 2, 1000)

    is_bitwise = f in (operator.and_, np.bitwise_and, operator.or_,
                       np.bitwise_or, operator.xor, np.bitwise_xor)

    if is_bitwise:
        p1 = test_utils.FuncAction(lambda t, l, o: f(round(t * 100), round(t * 123 + 1)))
        expect1 = [f(round(t * 100), round(t * 123 + 1)) for t in ts1]
    else:
        p1 = test_utils.FuncAction(lambda t, l, o: f(t, t * 2 + 1))
        expect1 = [f(t, t * 2 + 1) for t in ts1]

    test1 = test_utils.RampTest(p1, 2, 0)
    v1 = test1.eval_runtime(0, ts1)
    assert cmp_list(v1, expect1)

    for v in values:
        rv = test_utils.new_extern(lambda: v)
        if is_bitwise:
            p2 = test_utils.FuncAction(lambda t, l, o: f(rv, round(t * 123 + 1)))
            p3 = test_utils.FuncAction(lambda t, l, o: f(round(t * 100), rv))
        else:
            p2 = test_utils.FuncAction(lambda t, l, o: f(rv, t * 2 + 1))
            p3 = test_utils.FuncAction(lambda t, l, o: f(t, rv))
        @none_on_error
        def expect2():
            if is_bitwise:
                return [throw_non_finite(f, v, round(t * 123 + 1)) for t in ts1]
            else:
                return [throw_non_finite(f, v, t * 2 + 1) for t in ts1]
        @none_on_error
        def expect3():
            if is_bitwise:
                return [throw_non_finite(f, round(t * 100), v) for t in ts1]
            else:
                return [throw_non_finite(f, t, v) for t in ts1]
        test2 = test_utils.RampTest(p2, 2, 0)
        test3 = test_utils.RampTest(p3, 2, 0)
        v2 = none_on_error(test2.eval_runtime, 0, ts1)
        v3 = none_on_error(test3.eval_runtime, 0, ts1)
        assert cmp_list(v2, expect2)
        assert cmp_list(v3, expect3)

    ts = np.linspace(0, 2, 3)
    for v1 in values:
        rv1 = test_utils.new_extern(lambda: v1)
        for v2 in values:
            if isinstance(v1, bool) and isinstance(v2, bool) and f is np.subtract:
                # Numpy error that we don't want to follow
                continue
            if isinstance(v1, (bool, int)) and v2 < 0 and f is np.power:
                # Numpy error that we don't want to follow
                continue
            if v2 == 0 and (f is operator.mod or f is np.remainder):
                # Numpy doesn't raise an error in this case for some reason
                continue
            rv2 = test_utils.new_extern(lambda: v2)
            p4 = test_utils.FuncAction(lambda t, l, o: f(rv1, rv2))
            @none_on_error
            def expect4():
                fv = throw_non_finite(f, v1, v2)
                return [fv for t in ts]
            test4 = test_utils.RampTest(p4, 2, 0)
            v4 = none_on_error(test4.eval_runtime, 0, ts)
            assert cmp_list(v4, expect4)

def test_ops():
    olderr = np.seterr(all='ignore')
    for f in [operator.pos, operator.neg, rtval.inv, np.logical_not, abs, np.abs,
              math.ceil, np.ceil, np.exp, np.expm1, math.floor, np.floor, np.log,
              np.log1p, np.log2, np.log10, np.sqrt, np.arcsin, np.arccos, np.arctan,
              np.arcsinh, np.arccosh, np.arctanh, np.sin, np.cos, np.tan, np.sinh,
              np.cosh, np.tanh, np.rint, round, rtval.convert_bool]:
        run_check_unary(f)
    for f in [operator.add, operator.sub, np.subtract, operator.mul, np.multiply,
              operator.truediv, np.divide, operator.lt, np.less, operator.gt,
              np.greater, operator.le, np.less_equal, operator.ge, np.greater_equal,
              operator.ne, np.not_equal, operator.eq, np.equal,
              operator.and_, np.bitwise_and, operator.or_, np.bitwise_or,
              operator.xor, np.bitwise_xor, operator.pow, np.power, np.arctan2,
              np.hypot, np.fmin, np.fmax, operator.mod, np.remainder]:
        run_check_binary(f)
    np.seterr(**olderr)
