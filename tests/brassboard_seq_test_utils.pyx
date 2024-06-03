# cython: language_level=3

from brassboard_seq cimport action, event_time, rtval
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

def round_time(v):
    if rtval.is_rtval(v):
        return event_time.round_time_rt(v)
    else:
        return event_time.round_time_int(v)

def new_time_manager():
    return event_time.new_time_manager()

def time_manager_new_time(event_time.TimeManager time_manager,
                          event_time.EventTime prev, offset,
                          bint floating, cond, event_time.EventTime wait_for):
    if rtval.is_rtval(offset):
        return time_manager.new_time_rt(prev, offset, floating, cond, wait_for)
    else:
        return time_manager.new_time_int(prev, offset, floating, cond, wait_for)

def time_manager_new_round_time(event_time.TimeManager time_manager,
                                event_time.EventTime prev, offset,
                                bint floating, cond, event_time.EventTime wait_for):
    return time_manager.new_round_time(prev, offset, floating, cond, wait_for)

def time_manager_finalize(event_time.TimeManager time_manager):
    time_manager.finalize()

def time_manager_compute_all_times(event_time.TimeManager time_manager, unsigned age):
    max_time = time_manager.compute_all_times(age)
    ntimes = time_manager.time_values.size()
    values = []
    for i in range(ntimes):
        values.append(time_manager.time_values[i])
    return max_time, values

def time_manager_nchain(event_time.TimeManager time_manager):
    event_times = time_manager.event_times
    if len(event_times) == 0:
        return 0
    t = <event_time.EventTime>event_times[0]
    return t.chain_pos.size()

def event_time_set_base(event_time.EventTime self, event_time.EventTime base, offset):
    if rtval.is_rtval(offset):
        event_time.set_base_rt(self, base, offset)
    else:
        event_time.set_base_int(self, base, offset)

def event_time_id(event_time.EventTime self):
    return self.data.id

def event_time_get_static(event_time.EventTime self):
    return self.data.get_static()

def event_time_is_ordered(event_time.EventTime t1, event_time.EventTime t2):
    res = event_time.is_ordered(t1, t2)
    assert res != event_time.OrderError
    if res == event_time.NoOrder:
        assert event_time.is_ordered(t2, t1) == event_time.NoOrder
        return 'NoOrder'
    elif res == event_time.OrderBefore:
        assert event_time.is_ordered(t2, t1) == event_time.OrderAfter
        return 'OrderBefore'
    elif res == event_time.OrderEqual:
        assert event_time.is_ordered(t2, t1) == event_time.OrderEqual
        return 'OrderEqual'
    elif res == event_time.OrderAfter:
        assert event_time.is_ordered(t2, t1) == event_time.OrderBefore
        return 'OrderAfter'
    assert False
