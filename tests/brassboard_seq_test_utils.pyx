# cython: language_level=3

from brassboard_seq cimport action, event_time, rtval, seq, utils

import numpy as np

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cpython cimport PyObject, Py_INCREF

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
        std::vector<int> height(N <= 2 ? 0 : N - 2);
        brassboard_seq::order_to_rank(RK, SA);
        brassboard_seq::get_height_array(height, S, SA, RK);
        return height;
    }
    struct MaxRange {
        int i0;
        int i1;
        int maxv;
    };
    static inline std::vector<MaxRange> _get_max_range(std::vector<int> value)
    {
        std::vector<MaxRange> res;
        brassboard_seq::foreach_max_range(value, [&] (int i0, int i1, int maxv) {
            res.push_back({ i0, i1, maxv });
        });
        return res;
    }
    """
    vector[int] _get_suffix_array(vector[int])
    vector[int] _get_height_array(vector[int], vector[int])
    cppclass MaxRange:
        int i0
        int i1
        int maxv
    vector[MaxRange] _get_max_range(vector[int])

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
    return rtval.new_const(c, <rtval.RuntimeValue>None)

def new_arg(idx):
    return rtval.new_arg(idx)

cdef class Action:
    cdef unique_ptr[action.Action] tofree
    cdef action.Action *action
    cdef object ref

    def __str__(self):
        return action.action_str(self.action)

    def __repr__(self):
        return action.action_str(self.action)

cdef _ref_action(action.Action *p, parent):
    a = <Action>Action.__new__(Action)
    a.action = p
    a.ref = parent
    return a

def new_action(value, cond, bint is_pulse, bint exact_time, dict kws, int aid):
    a = <Action>Action.__new__(Action)
    cdef utils.py_object _kws
    if kws is not None:
        Py_INCREF(kws)
        _kws.reset(<PyObject*>kws)
    p = new action.Action(value, cond, is_pulse, exact_time, move(_kws), aid)
    p.length = <PyObject*>NULL
    a.action = p
    a.tofree.reset(p)
    return a

def action_set_tid(Action action, int tid):
    action.action.tid = tid

def action_get_aid(Action action):
    return action.action.aid

def action_get_is_pulse(Action action):
    return action.action.is_pulse

def action_get_exact_time(Action action):
    return action.action.exact_time

cdef get_pyobject(utils.py_object &v):
    p = v.get()
    if p == NULL:
        return
    return <object>p

def action_get_cond(Action action):
    return get_pyobject(action.action.cond)

def action_get_value(Action action):
    return get_pyobject(action.action.value)

def action_get_compile_info(Action action):
    pa = action.action
    return dict(tid=pa.tid, end_tid=pa.end_tid,
                length=<object>pa.length if pa.length != NULL else None,
                end_val=get_pyobject(pa.end_val))

def action_get_cond_val(Action action):
    return action.action.cond_val

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
        return event_time.round_time_rt(<rtval.RuntimeValue>v, event_time.rt_time_scale)
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
    cdef int nchn = len(s.seqinfo.channel_paths)
    all_actions = <vector[action.Action*]*>s.all_actions.get()
    res = []
    for cid in range(nchn):
        actions = all_actions[cid]
        res.append([_ref_action(action, s) for action in actions])
    return res

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

def get_max_range(list v):
    return [(mr.i0, mr.i1, mr.maxv) for mr in _get_max_range(v)]

def check_range(list vs, int i0, int i1, int maxv):
    cdef bint found_equal = False
    cdef int i
    cdef int v
    for i in range(i0, i1 + 1):
        v = vs[i]
        if v == maxv:
            found_equal = True
        elif v < maxv:
            return False
    return found_equal
