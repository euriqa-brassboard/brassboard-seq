# cython: language_level=3

from brassboard_seq cimport action
from cpython cimport PyObject

import numpy as np

def new_action(value, cond, bint is_pulse, bint exact_time, dict kws):
    return action.new_action(value, cond, is_pulse, exact_time, kws)

def action_set_tid(action.Action action, int tid):
    action.tid = tid

cdef class RampBufferTest:
    cdef action.RampFunction func
    cdef action.RampBuffer buff
    def __init__(self, func):
        self.func = func
        self.buff = action.new_ramp_buffer()

    def eval_compile(self, t, length, oldval):
        self.func.set_compile_params()
        return self.func.eval(t, length, oldval)

    def eval_runtime(self, age, ts, length, oldval):
        self.func.set_runtime_params(age)
        nt = len(ts)
        cdef double *buff_ptr = action.rampbuffer_alloc_input(<PyObject*>self.buff, nt)
        if buff_ptr == NULL:
            raise self.buff.exception
        for i in range(nt):
            buff_ptr[i] = ts[i]
        cdef double *output_ptr = action.rampbuffer_eval(<PyObject*>self.buff,
                                                         <PyObject*>self.func,
                                                         length, oldval)
        if output_ptr == NULL:
            raise self.buff.exception
        output = np.empty(nt)
        for i in range(nt):
            output[i] = output_ptr[i]
        return output
