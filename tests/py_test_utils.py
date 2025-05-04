#

from brassboard_seq.action import RampFunction
from brassboard_seq import rtval, utils

from brassboard_seq_test_utils import *

import pytest
import numpy as np

def with_seq_params(f):
    import inspect
    old_sig = inspect.signature(f)
    params = [inspect.Parameter('log_level', inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    params.extend(old_sig.parameters.values())
    new_sig = inspect.Signature(params)
    def cb(log_level, *args, **kws):
        try:
            utils.set_log_level(log_level)
            return f(*args, **kws)
        finally:
            utils.set_log_level("")
    cb.__name__ = f.__name__
    if hasattr(f, 'pytestmark'):
        cb.pytestmark = f.pytestmark
    cb.__signature__ = new_sig
    cb = pytest.mark.parametrize("max_bt", [0, 5, 500])(cb)
    return pytest.mark.parametrize("log_level", ["info", "debug", ""])(cb)

def check_bt(exc, max_bt, *names):
    fnames = [tb.name for tb in exc.traceback]
    for name in names:
        if max_bt == 0:
            assert name not in fnames
        else:
            assert name in fnames

class StaticFunction(RampFunction):
    def eval(self, t, length, oldval):
        return t / 2 + oldval - length

class SinFunction(RampFunction):
    def __init__(self, amp, freq, phase):
        super().__init__(amp=amp, freq=freq, phase=phase)

    def eval(self, t, length, oldval):
        return oldval + self.amp * np.sin(self.freq * t + self.phase)

class ValueFunction(RampFunction):
    def __init__(self, res):
        self.res = res
        super().__init__()

    def eval(self, t, length, oldval):
        return self.res

class ErrorFunction(RampFunction):
    def __init__(self, err):
        self.err = err
        super().__init__()

    def eval(self, t, length, oldval):
        raise self.err

class ErrorEndFunction(RampFunction):
    def __init__(self, err):
        self.err = err
        super().__init__()

    def eval(self, t, length, oldval):
        if t is length:
            raise self.err
        return t / length

class PyCubicSpline(RampFunction):
    def __init__(self, order0, order1=0.0, order2=0.0, order3=0.0):
        super().__init__(order0=order0, order1=order1, order2=order2, order3=order3)

    def eval(self, t, length, oldval):
        t = t / length
        return self.order0 + (self.order1 + (self.order2 + self.order3 * t) * t) * t

    def spline_segments(self, length, oldval):
        return ()

class FuncAction(RampFunction):
    def __init__(self, cb):
        self.cb = cb
        super().__init__()

    def eval(self, t, l, o):
        return self.cb(t, l, o)

class LinearRampNoSeg(RampFunction):
    def __init__(self, start, end):
        super().__init__(start=start, end=end)

    def eval(self, t, length, oldval):
        t = t / length
        return self.start * (1 - t) + self.end * t

class RampUpAndDown(RampFunction):
    def __init__(self, pos, val):
        super().__init__(pos=pos, val=val)

    def eval(self, t, length, oldval):
        t = t / length
        len1 = self.pos
        len2 = 1 - self.pos
        t1 = t / len1
        t2 = (t - self.pos) / len2
        v1 = oldval * (1 - t1) + self.val * t1
        v2 = self.val * (1 - t2) + oldval * t2
        return rtval.ifelse(t > self.pos, v2, v1)

    def spline_segments(self, length, oldval):
        return (length * self.pos,)

class ErrorSegment(RampFunction):
    def eval(self, t, length, oldval):
        return t

    def spline_segments(self, length, oldval):
        raise ValueError("JJJLLL---DFFDAFD")

class CustomSegment(RampFunction):
    def __init__(self, seg):
        self.seg = seg
        super().__init__()

    def eval(self, t, length, oldval):
        return t

    def spline_segments(self, length, oldval):
        return self.seg

class DivLengthFunction(RampFunction):
    def eval(self, t, length, oldval):
        return t / length
