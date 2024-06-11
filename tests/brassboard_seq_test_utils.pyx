# cython: language_level=3

from brassboard_seq cimport action, rtval
from cpython cimport PyObject

import numpy as np

def new_invalid_rtval():
    # This should only happen if something really wrong happens.
    # We'll just test that we behave reasonably enough.
    # (it's unavoidable that we'll crash in some cases)
    rv = rtval._new_rtval(<rtval.ValueType>1000)
    rv.arg0 = rtval.new_const(1)
    rv.arg1 = rtval.new_const(1)
    return rv

def new_action(value, cond, bint is_pulse, bint exact_time, dict kws, int aid):
    return action.new_action(value, cond, is_pulse, exact_time, kws, aid)

def action_set_tid(action.Action action, int tid):
    action.tid = tid

def action_get_aid(action.Action action):
    return action.aid

cdef class RampBufferTest:
    cdef action.RampFunction func
    cdef action.RampBuffer buff
    def __init__(self, func):
        self.func = func
        self.buff = action.new_ramp_buffer()

    def eval_compile(self, t, length, oldval):
        action.ramp_set_compile_params(self.func)
        return action.ramp_eval(self.func, t, length, oldval)

    def eval_runtime(self, age, ts, length, oldval):
        action.ramp_set_runtime_params(self.func, age)
        nt = len(ts)
        cdef double *buff_ptr = action.rampbuffer_alloc_input(self.buff, nt)
        for i in range(nt):
            buff_ptr[i] = ts[i]
        cdef double *output_ptr = action.rampbuffer_eval(self.buff, self.func,
                                                         length, oldval)
        output = np.empty(nt)
        for i in range(nt):
            output[i] = output_ptr[i]
        return output
