# cython: language_level=3

from brassboard_seq cimport action, event_time, rtval, seq, utils

import numpy as np

from libcpp.vector cimport vector

cdef extern from *:
    """
    #include <vector>

    static inline std::vector<int> _get_suffix_array(std::vector<int> S)
    {
        int N = S.size();
        std::vector<int> SA(N);
        std::vector<int> ws(N);
        brassboard_seq::get_suffix_array(SA, S, ws);
        return SA;
    }
    static inline std::vector<int> _get_height_array(std::vector<int> S,
                                                     std::vector<int> SA)
    {
        int N = S.size();
        std::vector<int> RK(N);
        std::vector<int> height(N);
        brassboard_seq::order_to_rank(RK, SA);
        brassboard_seq::get_height_array(height, S, SA, RK);
        return height;
    }
    """
    vector[int] _get_suffix_array(vector[int])
    vector[int] _get_height_array(vector[int], vector[int])

def new_invalid_rtval():
    # This should only happen if something really wrong happens.
    # We'll just test that we behave reasonably enough.
    # (it's unavoidable that we'll crash in some cases)
    rv = <rtval.RuntimeValue>rtval.RuntimeValue.__new__(rtval.RuntimeValue)
    rv.type_ = <rtval.ValueType>1000
    rv.datatype = rtval.DataType.Float64
    rv.age = -1
    rv.arg0 = new_const(1)
    rv.arg1 = new_const(1)
    return rv

def new_const(c):
    return rtval.new_const(rtval.RuntimeValue, c, <rtval.RuntimeValue>None)

def new_arg(idx):
    return rtval.new_arg(idx)

def new_action(value, cond, bint is_pulse, bint exact_time, dict kws, int aid):
    return action.new_action(action.Action, value, cond, is_pulse, exact_time,
                             kws, aid, None)

def action_set_tid(action.Action action, int tid):
    action.tid = tid

def action_get_aid(action.Action action):
    return action.aid

def action_get_is_pulse(action.Action action):
    return action.data.is_pulse

def action_get_exact_time(action.Action action):
    return action.data.exact_time

def action_get_cond(action.Action action):
    return action.cond

def action_get_value(action.Action action):
    return action.value

def action_get_compile_info(action.Action action):
    return dict(tid=action.tid, end_tid=action.end_tid, length=action.length,
                prev_val=action.prev_val, end_val=action.end_val)

def action_get_cond_val(action.Action action):
    return action.data.cond_val

cdef double tagval_to_float(rtval.TagVal tv):
    rtval.throw_py_error(tv.err)
    return tv.val.f64_val

cdef class RampTest:
    cdef action.RampFunction func
    cdef object length
    cdef object oldval

    def __init__(self, func, length, oldval):
        self.func = func
        self.length = length
        self.oldval = oldval
        self.func.set_compile_params(length, oldval)

    def eval_compile(self, t):
        return self.func.eval(t, self.length, self.oldval)

    def eval_runtime(self, age, ts):
        cdef utils.py_object pyage
        self.func.set_runtime_params(age, pyage)
        return [tagval_to_float(self.func.runtime_eval(t)) for t in ts]

def ramp_get_spline_segments(action.RampFunction self, length, oldval):
    return self.spline_segments(length, oldval)

def round_time(v):
    if rtval.is_rtval(v):
        return event_time.round_time_rt(rtval.RuntimeValue, <rtval.RuntimeValue>v,
                                        event_time.rt_time_scale)
    else:
        return event_time.round_time_int(v)

def new_time_manager():
    return event_time.new_time_manager()

def time_manager_new_time(event_time.TimeManager time_manager,
                          event_time.EventTime prev, offset,
                          bint floating, cond, event_time.EventTime wait_for):
    if rtval.is_rtval(offset):
        assert not floating
        return time_manager.new_time_rt(prev, offset, cond, wait_for)
    else:
        return time_manager.new_time_int(prev, offset, floating, cond, wait_for)

def time_manager_new_round_time(event_time.TimeManager time_manager,
                                event_time.EventTime prev, offset,
                                cond, event_time.EventTime wait_for):
    return time_manager.new_round_time(prev, offset, cond, wait_for)

def time_manager_finalize(event_time.TimeManager time_manager):
    time_manager.finalize()

def time_manager_compute_all_times(event_time.TimeManager time_manager, unsigned age):
    cdef utils.py_object pyage
    max_time = time_manager.compute_all_times(age, pyage)
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

def seq_get_channel_paths(seq.Seq s):
    return s.seqinfo.channel_paths

def seq_get_event_time(seq.Seq s, int tid):
    return s.seqinfo.time_mgr.event_times[tid]

def seq_get_cond(s):
    if type(s) is seq.ConditionalWrapper:
        return (<seq.ConditionalWrapper>s).cond
    return (<seq.TimeSeq?>s).cond

def seq_finalize(seq.Seq s):
    s.finalize()

def seq_get_all_actions(seq.Seq s):
    return s.all_actions

def seq_runtime_finalize(seq.Seq s, unsigned age):
    cdef utils.py_object pyage
    s.runtime_finalize(age, pyage)

def seq_get_all_times(seq.Seq s):
    time_mgr = s.seqinfo.time_mgr
    ntimes = time_mgr.time_values.size()
    values = []
    for i in range(ntimes):
        values.append(time_mgr.time_values[i])
    return s.total_time, values

def get_suffix_array(ary):
    return _get_suffix_array(ary)

def get_height_array(s, sa):
    return _get_height_array(s, sa)
