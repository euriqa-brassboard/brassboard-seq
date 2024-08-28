#

from brassboard_seq import action, rtval
import brassboard_seq_test_utils as test_utils
import pytest

import numpy as np

def check_action_str(action, s):
    assert str(action) == s
    assert repr(action) == s

def test_action():
    with pytest.raises(TypeError):
        action.Action()
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

    assert a1 < a2
    assert a2 > a1
    assert a1 != a2
    assert not (a1 == a2)
    assert a1 == a1

    assert a1 < a3
    assert a3 > a1

    assert a2 > a3
    assert a3 < a2

    with pytest.raises(TypeError):
        a1 <= a2
    with pytest.raises(TypeError):
        a2 >= a1
    with pytest.raises(TypeError):
        a2 >= 1

    assert (a2 == 1) is False
    assert (a2 != 0) is True

class StaticFunction(action.RampFunction):
    def __init__(self):
        action.RampFunction.__init__(self)

    def eval(self, t, length, oldval):
        return t / 2 + oldval - length

class SinFunction(action.RampFunction):
    def __init__(self, amp, freq, phase):
        action.RampFunction.__init__(self, amp=amp, freq=freq, phase=phase)

    def eval(self, t, length, oldval):
        return oldval + self.amp * np.sin(self.freq * t + self.phase)

class WrongValueFunction(action.RampFunction):
    def __init__(self, res):
        action.RampFunction.__init__(self)
        self.res = res

    def eval(self, t, length, oldval):
        return self.res

class ErrorFunction(action.RampFunction):
    def __init__(self, err):
        action.RampFunction.__init__(self)
        self.err = err

    def eval(self, t, length, oldval):
        raise self.err

class PyCubicSpline(action.RampFunction):
    def __init__(self, order0, order1=0.0, order2=0.0, order3=0.0):
        super().__init__(order0=order0, order1=order1, order2=order2, order3=order3)
    def eval(self, t, length, oldval):
        t = t / length
        return self.order0 + (self.order1 + (self.order2 + self.order3 * t) * t) * t
    def spline_segments(self, length, oldval):
        return ()

def test_rampbuffer():
    with pytest.raises(TypeError):
        action.RampBuffer()
    with pytest.raises(AttributeError):
        # Subclass of RampFunction must have eval defined.
        action.RampFunction()

    rt = rtval.new_extern(lambda: 1)
    rlen = rtval.new_extern(lambda: 2)
    rold = rtval.new_extern(lambda: -0.2)
    st = str(rt)
    slen = str(rlen)
    sold = str(rold)

    test = test_utils.RampBufferTest(StaticFunction())
    rv = test.eval_compile(rt, rlen, rold)
    assert str(rv) == f'({st} / 2 + {sold}) - {slen}'
    assert rtval.get_value(rv, 0) == 1 / 2 + -0.2 - 2

    ts = np.linspace(-1, 1, 1000)
    vs = test.eval_runtime(0, ts, 2, -0.2)
    assert (vs == ts / 2 + -0.2 - 2).all()

    ts = np.linspace(-1, 1, 100)
    vs = test.eval_runtime(0, ts, 2, -0.2)
    assert (vs == ts / 2 + -0.2 - 2).all()

    ts = np.linspace(-1, 1, 2000)
    vs = test.eval_runtime(0, ts, 2, -0.2)
    assert (vs == ts / 2 + -0.2 - 2).all()

    ramp = rtval.new_extern(lambda: 2)
    rfreq = rtval.new_extern(lambda: 1.2)
    rphase = 2.3
    samp = str(ramp)
    sfreq = str(rfreq)
    test = test_utils.RampBufferTest(SinFunction(ramp, rfreq, rphase))
    rv = test.eval_compile(rt, rlen, rold)
    assert str(rv) == f'{sold} + {samp} * sin(2.3 + {sfreq} * {st})'
    assert rtval.get_value(rv, 0) == -0.2 + 2 * np.sin(2.3 + 1.2 * 1)

    ts = np.linspace(-1, 1, 1000)
    vs = test.eval_runtime(0, ts, 2, -0.2)
    assert (vs == -0.2 + 2 * np.sin(2.3 + 1.2 * ts)).all()

    ts = np.linspace(-1, 1, 100)
    vs = test.eval_runtime(0, ts, 2, -0.2)
    assert (vs == -0.2 + 2 * np.sin(2.3 + 1.2 * ts)).all()

    ts = np.linspace(-1, 1, 2000)
    vs = test.eval_runtime(0, ts, 2, -0.2)
    assert (vs == -0.2 + 2 * np.sin(2.3 + 1.2 * ts)).all()

    wtfunc = WrongValueFunction([])
    test = test_utils.RampBufferTest(wtfunc)
    with pytest.raises(TypeError):
        test.eval_runtime(0, ts, 2, -0.2)
    wtfunc.res = 1
    with pytest.raises(TypeError):
        test.eval_runtime(0, ts, 2, -0.2)
    wtfunc.res = np.empty((2, 2))
    with pytest.raises(ValueError):
        test.eval_runtime(0, ts, 2, -0.2)

    wtfunc.res = np.empty(len(ts))
    test.eval_runtime(0, ts, 2, -0.2)
    wtfunc.res = np.empty(len(ts) + 2)
    with pytest.raises(ValueError):
        test.eval_runtime(0, ts, 2, -0.2)

    # Test type conversion
    wtfunc.res = np.random.randint(1000, size=len(ts))
    assert (test.eval_runtime(0, ts, 2, -0.2) == wtfunc.res).all()

    efunc = ErrorFunction(ValueError("AAAAA"))
    test = test_utils.RampBufferTest(efunc)
    with pytest.raises(ValueError, match="^AAAAA$"):
        test.eval_compile(rt, rlen, rold)
    with pytest.raises(ValueError, match="^AAAAA$"):
        test.eval_runtime(0, ts, 2, -0.2)

def test_spline():
    assert test_utils.ramp_get_spline_segments(SinFunction(1.0, 2.0, 0.1), 1, 0) is None

    o0 = 0.1
    o1 = 0.2
    o2 = 1
    o3 = 0.3

    def check_spline(rt0, rt1, rt2, rt3):
        ro0 = rtval.new_extern(lambda: o0) if rt0 else o0
        ro1 = rtval.new_extern(lambda: o1) if rt1 else o1
        ro2 = rtval.new_extern(lambda: o2) if rt2 else o2
        ro3 = rtval.new_extern(lambda: o3) if rt3 else o3

        sp_seq = action.SeqCubicSpline(ro0, ro1, ro2, ro3)
        assert sp_seq.order0 is ro0
        assert sp_seq.order1 is ro1
        assert sp_seq.order2 is ro2
        assert sp_seq.order3 is ro3
        sp_py = PyCubicSpline(ro0, ro1, ro2, ro3)
        assert test_utils.ramp_get_spline_segments(sp_seq, 1, 0) == ()
        assert test_utils.ramp_get_spline_segments(sp_py, 1, 0) == ()

        test_seq = test_utils.RampBufferTest(sp_seq)
        test_py = test_utils.RampBufferTest(sp_py)

        v0_seq = test_seq.eval_compile(0, 1, 0.2)
        v05_seq = test_seq.eval_compile(0.5, 1, 0.2)
        v1_seq = test_seq.eval_compile(1, 1, 0.2)
        assert rtval.get_value(v0_seq, 1) == 0.1
        assert rtval.get_value(v05_seq, 1) == pytest.approx(0.4875)
        assert rtval.get_value(v1_seq, 1) == 1.6

        v0_py = test_py.eval_compile(0, 1, 0.2)
        v05_py = test_py.eval_compile(0.5, 1, 0.2)
        v1_py = test_py.eval_compile(1, 1, 0.2)
        assert rtval.get_value(v0_py, 1) == 0.1
        assert rtval.get_value(v05_py, 1) == pytest.approx(0.4875)
        assert rtval.get_value(v1_py, 1) == 1.6

        assert list(test_seq.eval_runtime(2, [0, 0.5, 1], 1, 10)) == pytest.approx([0.1, 0.4875, 1.6])
        assert list(test_py.eval_runtime(2, [0, 0.5, 1], 1, 10)) == pytest.approx([0.1, 0.4875, 1.6])

    for i in range(15):
        check_spline(i & 1, i & 2, i & 4, i & 8)

def blackman_func(t): # t in [0, 1]
    theta = (t - 0.5) * 2 * np.pi
    return 21/50 + 1/2 * np.cos(theta) + 2/25 * np.cos(2 * theta)

def test_blackman():
    test1 = test_utils.RampBufferTest(action.Blackman(0.9, 0.1))
    test2 = test_utils.RampBufferTest(action.BlackmanSquare(0.6, -0.1))
    ts = np.linspace(0, 1, 1000)
    v1 = list(test1.eval_runtime(0, ts, 1, 0))
    v2 = list(test2.eval_runtime(0, ts, 1, 0))
    bs = blackman_func(ts)
    expect1 = list(0.1 + 0.9 * bs)
    expect2 = list(-0.1 + 0.6 * bs**2)
    assert v1 == pytest.approx(expect1)
    assert v2 == pytest.approx(expect2)
