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
