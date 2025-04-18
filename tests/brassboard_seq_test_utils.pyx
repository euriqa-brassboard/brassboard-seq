# cython: language_level=3

from brassboard_seq cimport action, backend, event_time, rtval, seq, utils, yaml

import numpy as np

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from libcpp.string cimport string as cppstr

from libc.stdint cimport *

from cpython cimport PyObject, Py_INCREF, Py_EQ, Py_NE, Py_GT, Py_LT, Py_GE, Py_LE, \
  PyBytes_GET_SIZE, PyBytes_AS_STRING, PyBytes_FromStringAndSize

cdef extern from "test_utils.cpp" namespace "brassboard_seq":
    char *to_chars(char[], int) except +
    int throw_if_not(int) except +
    int throw_if(int) except +

    rtval.TagVal test_callback_extern(TestCallback) except +
    rtval.TagVal test_callback_extern_age(TestCallback, unsigned) except +
    rtval.RuntimeValue _new_invalid_rtval() except +
    vector[int] _get_suffix_array(vector[int])
    vector[int] _get_height_array(vector[int], vector[int])
    cppclass MaxRange:
        int i0
        int i1
        int maxv
    vector[MaxRange] _get_max_range(vector[int])
    cppclass _Bits_i32x5:
        _Bits_i32x5()
        _Bits_i32x5(int32_t)
        int32_t &operator[](int)
        _Bits_i32x5 operator<<(int) const
        _Bits_i32x5 operator>>(int) const
        bint operator==(const _Bits_i32x5&) const
        bint operator!=(const _Bits_i32x5&) const
        bint operator<(const _Bits_i32x5&) const
        bint operator>(const _Bits_i32x5&) const
        bint operator<=(const _Bits_i32x5&) const
        bint operator>=(const _Bits_i32x5&) const

    cppclass _Bits_i64x4:
        _Bits_i64x4()
        _Bits_i64x4(int64_t)
        int64_t &operator[](int)
        _Bits_i64x4 operator<<(int) const
        _Bits_i64x4 operator>>(int) const
        bint operator==(const _Bits_i64x4&) const
        bint operator!=(const _Bits_i64x4&) const
        bint operator<(const _Bits_i64x4&) const
        bint operator>(const _Bits_i64x4&) const
        bint operator<=(const _Bits_i64x4&) const
        bint operator>=(const _Bits_i64x4&) const

    cppclass _Bits_u64x4:
        _Bits_u64x4()
        _Bits_u64x4(uint64_t)
        uint64_t &operator[](int)
        _Bits_u64x4 operator<<(int) const
        _Bits_u64x4 operator>>(int) const
        bint operator==(const _Bits_u64x4&) const
        bint operator!=(const _Bits_u64x4&) const
        bint operator<(const _Bits_u64x4&) const
        bint operator>(const _Bits_u64x4&) const
        bint operator<=(const _Bits_u64x4&) const
        bint operator>=(const _Bits_u64x4&) const

    cppclass _Bits_i8x43:
        _Bits_i8x43()
        _Bits_i8x43(int8_t)
        int8_t &operator[](int)
        _Bits_i8x43 operator<<(int) const
        _Bits_i8x43 operator>>(int) const
        bint operator==(const _Bits_i8x43&) const
        bint operator!=(const _Bits_i8x43&) const
        bint operator<(const _Bits_i8x43&) const
        bint operator>(const _Bits_i8x43&) const
        bint operator<=(const _Bits_i8x43&) const
        bint operator>=(const _Bits_i8x43&) const

    T get_mask[T](unsigned b1, unsigned b2)
    Res convert[Res,In](const In&)
    T op_not[T](const T &v)
    Res op_or[Res,T1,T2](const T1&, const T2&)
    Res op_and[Res,T1,T2](const T1&, const T2&)
    Res op_xor[Res,T1,T2](const T1&, const T2&)
    void op_ior[T1,T2](T1&, const T2&)
    void op_iand[T1,T2](T1&, const T2&)
    void op_ixor[T1,T2](T1&, const T2&)
    bint bits_to_bool[T](const T&)
    object bits_to_pylong[T](const T&)
    object bits_to_pybytes[T](const T&)
    cppstr bits_to_str[T](const T&, bint)
    cppclass test_istream:
        test_istream &seekg(ssize_t)
        test_istream &seekg2 "seekg"(ssize_t, utils.seekdir)
        bint fail() const
    cppclass test_istream_ba:
        test_istream_ba &seekg(ssize_t)
        test_istream_ba &seekg2 "seekg"(ssize_t, utils.seekdir)
        bint fail() const

    event_time.TimeManager _new_time_manager() except +
    event_time.EventTime timemanager_new_round_time(event_time.TimeManager, event_time.EventTime, object, object, event_time.EventTime) except +

    event_time.EventTime timemanager_new_time_int(event_time.TimeManager self, event_time.EventTime prev, int64_t offset, bint floating, object cond, event_time.EventTime wait_for) except +
    rtval.rtval_ref round_time_rt "brassboard_seq::event_time::round_time_rt" (rtval.RuntimeValue) except +

    object condseq_get_cond(object) except +

    void init_action_obj(Action action, object value, object cond, bint is_pulse,
                         bint exact_time, object kws, int aid) except +
    object _action_get_cond(action.Action *action)
    object _action_get_value(action.Action *action)
    object _action_get_length(action.Action *action)
    object _action_get_end_val(action.Action *action)
    vector[action.Action*] *compiledseq_get_all_actions(backend.CompiledSeq &cseq)
    int64_t compiledseq_get_total_time(backend.CompiledSeq &cseq)
    void py_check_num_arg "brassboard_seq::py::check_num_arg" (
        const char *func_name, ssize_t nfound, ssize_t nmin, ssize_t nmax) except +
    void _timemanager_finalize(event_time.TimeManager) except +
    int64_t _timemanager_compute_all_times(event_time.TimeManager, unsigned) except +
    event_time.EventTime new_time_rt(event_time.TimeManager self,
                                     event_time.EventTime prev,
                                     rtval.RuntimeValue offset, object cond,
                                     event_time.EventTime wait_for) except +
    void event_time_set_base_int(event_time.EventTime self, event_time.EventTime base,
                                 int64_t offset) except +
    void event_time_set_base_rt(event_time.EventTime self, event_time.EventTime base,
                                rtval.RuntimeValue offset) except +
    void _yaml_io_print "brassboard_seq::yaml::print" (utils.stringio &io, object, int indent) except +

def new_invalid_rtval():
    # This should only happen if something really wrong happens.
    # We'll just test that we behave reasonably enough.
    # (it's unavoidable that we'll crash in some cases)
    rt = rtval.new_expr2(rtval.ValueType.Add, rtval.new_const(1).rel(), rtval.new_const(1).rel()).rel()
    rt.type_ = <rtval.ValueType>1000
    return rt

cdef class TestCallback(rtval.ExternCallback):
    cdef object cb
    cdef bint has_age
    def __str__(self):
        return f'extern_age({self.cb})' if self.has_age else f'extern({self.cb})'

cdef TestCallback new_test_callback(cb, bint has_age):
    self = <TestCallback>TestCallback.__new__(TestCallback)
    self.cb = cb
    self.has_age = has_age
    if has_age:
        self.fptr = <void*><rtval.TagVal(*)(TestCallback,unsigned)>test_callback_extern_age
    else:
        self.fptr = <void*><rtval.TagVal(*)(TestCallback)>test_callback_extern
    return self

def new_const(c):
    return rtval.new_const(c).rel()

def new_arg(idx):
    return rtval.new_arg(idx, float)

def new_extern_age(cb, ty=float):
    return rtval.new_extern_age(new_test_callback(cb, True), ty)

def new_extern(cb, ty=float):
    return rtval.new_extern(new_test_callback(cb, False), ty)

cdef class Action:
    cdef unique_ptr[action.Action] tofree
    cdef action.Action *action
    cdef object ref

    def __str__(self):
        return self.action.py_str()

    def __repr__(self):
        return self.action.py_str()

cdef _ref_action(action.Action *p, parent):
    a = <Action>Action.__new__(Action)
    a.action = p
    a.ref = parent
    return a

def new_action(value, cond, bint is_pulse, bint exact_time, dict kws, int aid):
    a = <Action>Action.__new__(Action)
    init_action_obj(a, value, cond, is_pulse, exact_time, kws, aid)
    return a

def action_set_tid(Action action, int tid):
    action.action.tid = tid

def action_get_aid(Action action):
    return action.action.aid

def action_get_is_pulse(Action action):
    return action.action.is_pulse

def action_get_exact_time(Action action):
    return action.action.exact_time

def action_get_cond(Action action):
    return _action_get_cond(action.action)

def action_get_value(Action action):
    return _action_get_value(action.action)

def action_get_compile_info(Action action):
    pa = action.action
    return dict(tid=pa.tid, end_tid=pa.end_tid,
                length=_action_get_length(pa),
                end_val=_action_get_end_val(pa))

def action_get_cond_val(Action action):
    return action.action.cond_val

cdef double tagval_to_float(rtval.TagVal tv):
    rtval.throw_py_error(tv.err)
    return tv.val.f64_val

cdef class RampTest:
    cdef action._RampFunctionBase func
    cdef object length
    cdef object oldval

    def __init__(self, func, length, oldval):
        self.func = func
        self.length = length
        self.oldval = oldval

    def eval_compile_end(self):
        return self.func.eval_end(self.length, self.oldval)

    def eval_runtime(self, unsigned age, ts):
        self.func.set_runtime_params(age)
        self.func.spline_segments(rtval.get_value_f64(self.length, age),
                                  rtval.get_value_f64(self.oldval, age))
        return [tagval_to_float(self.func.runtime_eval(t)) for t in ts]

def ramp_get_spline_segments(action._RampFunctionBase self, length, oldval):
    return self.spline_segments(length, oldval)

def ramp_runtime_eval(action._RampFunctionBase self, t):
    return tagval_to_float(self.runtime_eval(t))

def round_time(v):
    if rtval.is_rtval(v):
        return round_time_rt(<rtval.RuntimeValue>v).rel()
    else:
        return event_time.round_time_int(v)

def new_time_manager():
    return _new_time_manager()

def time_manager_new_time(event_time.TimeManager time_manager,
                          event_time.EventTime prev, offset,
                          bint floating, cond, event_time.EventTime wait_for):
    if rtval.is_rtval(offset):
        assert not floating
        return new_time_rt(time_manager, prev, offset, cond, wait_for)
    else:
        return timemanager_new_time_int(time_manager, prev, offset, floating, cond, wait_for)

def time_manager_new_round_time(event_time.TimeManager time_manager,
                                event_time.EventTime prev, offset,
                                cond, event_time.EventTime wait_for):
    return timemanager_new_round_time(time_manager, prev, offset, cond, wait_for)

def time_manager_finalize(event_time.TimeManager time_manager):
    _timemanager_finalize(time_manager)

def time_manager_compute_all_times(event_time.TimeManager time_manager, unsigned age):
    max_time = _timemanager_compute_all_times(time_manager, age)
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
        event_time_set_base_rt(self, base, offset)
    else:
        event_time_set_base_int(self, base, offset)

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
    return condseq_get_cond(s)

def compiler_get_all_actions(backend.SeqCompiler comp):
    s = comp.seq
    cdef int nchn = len(s.seqinfo.channel_paths)
    all_actions = compiledseq_get_all_actions(comp.cseq)
    res = []
    for cid in range(nchn):
        actions = all_actions[cid]
        res.append([_ref_action(action, s) for action in actions])
    return res

def compiler_get_all_times(backend.SeqCompiler comp):
    s = comp.seq
    time_mgr = s.seqinfo.time_mgr
    ntimes = time_mgr.time_values.size()
    values = []
    for i in range(ntimes):
        values.append(time_mgr.time_values[i])
    return compiledseq_get_total_time(comp.cseq), values

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

cdef class Bits_i32x5:
    cdef _Bits_i32x5 bits
    @staticmethod
    cdef Bits_i32x5 _new(_Bits_i32x5 bits):
        res = <Bits_i32x5>Bits_i32x5.__new__(Bits_i32x5)
        res.bits = bits
        return res

    def __init__(self, bits=0):
        if isinstance(bits, int):
            self.bits = _Bits_i32x5(<int32_t>bits)
        elif isinstance(bits, Bits_i32x5):
            self.bits = (<Bits_i32x5>bits).bits
        elif isinstance(bits, Bits_i64x4):
            self.bits = convert[_Bits_i32x5,_Bits_i64x4]((<Bits_i64x4>bits).bits)
        elif isinstance(bits, Bits_u64x4):
            self.bits = convert[_Bits_i32x5,_Bits_u64x4]((<Bits_u64x4>bits).bits)
        elif isinstance(bits, Bits_i8x43):
            self.bits = convert[_Bits_i32x5,_Bits_i8x43]((<Bits_i8x43>bits).bits)
        else:
            raise TypeError("Unknown input type")

    @staticmethod
    def spec():
        return 5, 32, True

    @staticmethod
    def get_mask(unsigned b1, unsigned b2):
        return Bits_i32x5._new(get_mask[_Bits_i32x5](b1, b2))

    def __richcmp__(self, Bits_i32x5 other, int op):
        if op == Py_EQ:
            return self.bits == other.bits
        if op == Py_NE:
            return self.bits != other.bits
        if op == Py_GT:
            return self.bits > other.bits
        if op == Py_LT:
            return self.bits < other.bits
        if op == Py_GE:
            return self.bits >= other.bits
        if op == Py_LE:
            return self.bits <= other.bits

    def __len__(self):
        return 5

    def __getitem__(self, int i):
        return (<const _Bits_i32x5&>self.bits)[i]

    def __setitem__(self, int i, v):
        self.bits[i] = v

    def __lshift__(self, int i):
        return Bits_i32x5._new(self.bits << i)

    def __rshift__(self, int i):
        return Bits_i32x5._new(self.bits >> i)

    def __index__(self):
        return bits_to_pylong(self.bits)

    def __bool__(self):
        return bits_to_bool(self.bits)

    def bytes(self):
        return bits_to_pybytes(self.bits)

    def __str__(self):
        return bytes(bits_to_str(self.bits, False)).decode()

    def __repr__(self):
        return bytes(bits_to_str(self.bits, True)).decode()

    def __invert__(self):
        return Bits_i32x5._new(op_not(self.bits))

    def __or__(self, other):
        if isinstance(other, Bits_i32x5):
            return Bits_i32x5._new(op_or[_Bits_i32x5,_Bits_i32x5,_Bits_i32x5](
                self.bits, (<Bits_i32x5>other).bits))
        elif isinstance(other, Bits_i64x4):
            return Bits_i64x4._new(op_or[_Bits_i64x4,_Bits_i32x5,_Bits_i64x4](
                self.bits, (<Bits_i64x4>other).bits))
        elif isinstance(other, Bits_u64x4):
            return Bits_u64x4._new(op_or[_Bits_u64x4,_Bits_i32x5,_Bits_u64x4](
                self.bits, (<Bits_u64x4>other).bits))
        elif isinstance(other, Bits_i8x43):
            return Bits_i8x43._new(op_or[_Bits_i8x43,_Bits_i32x5,_Bits_i8x43](
                self.bits, (<Bits_i8x43>other).bits))
        else:
            raise TypeError("Unknown input type")

    def __and__(self, other):
        if isinstance(other, Bits_i32x5):
            return Bits_i32x5._new(op_and[_Bits_i32x5,_Bits_i32x5,_Bits_i32x5](
                self.bits, (<Bits_i32x5>other).bits))
        elif isinstance(other, Bits_i64x4):
            return Bits_i64x4._new(op_and[_Bits_i64x4,_Bits_i32x5,_Bits_i64x4](
                self.bits, (<Bits_i64x4>other).bits))
        elif isinstance(other, Bits_u64x4):
            return Bits_u64x4._new(op_and[_Bits_u64x4,_Bits_i32x5,_Bits_u64x4](
                self.bits, (<Bits_u64x4>other).bits))
        elif isinstance(other, Bits_i8x43):
            return Bits_i8x43._new(op_and[_Bits_i8x43,_Bits_i32x5,_Bits_i8x43](
                self.bits, (<Bits_i8x43>other).bits))
        else:
            raise TypeError("Unknown input type")

    def __xor__(self, other):
        if isinstance(other, Bits_i32x5):
            return Bits_i32x5._new(op_xor[_Bits_i32x5,_Bits_i32x5,_Bits_i32x5](
                self.bits, (<Bits_i32x5>other).bits))
        elif isinstance(other, Bits_i64x4):
            return Bits_i64x4._new(op_xor[_Bits_i64x4,_Bits_i32x5,_Bits_i64x4](
                self.bits, (<Bits_i64x4>other).bits))
        elif isinstance(other, Bits_u64x4):
            return Bits_u64x4._new(op_xor[_Bits_u64x4,_Bits_i32x5,_Bits_u64x4](
                self.bits, (<Bits_u64x4>other).bits))
        elif isinstance(other, Bits_i8x43):
            return Bits_i8x43._new(op_xor[_Bits_i8x43,_Bits_i32x5,_Bits_i8x43](
                self.bits, (<Bits_i8x43>other).bits))
        else:
            raise TypeError("Unknown input type")

    def __ior__(self, other):
        if isinstance(other, Bits_i32x5):
            op_ior(self.bits, (<Bits_i32x5>other).bits)
        elif isinstance(other, Bits_i64x4):
            op_ior(self.bits, (<Bits_i64x4>other).bits)
        elif isinstance(other, Bits_u64x4):
            op_ior(self.bits, (<Bits_u64x4>other).bits)
        elif isinstance(other, Bits_i8x43):
            op_ior(self.bits, (<Bits_i8x43>other).bits)
        else:
            raise TypeError("Unknown input type")
        return self

    def __iand__(self, other):
        if isinstance(other, Bits_i32x5):
            op_iand(self.bits, (<Bits_i32x5>other).bits)
        elif isinstance(other, Bits_i64x4):
            op_iand(self.bits, (<Bits_i64x4>other).bits)
        elif isinstance(other, Bits_u64x4):
            op_iand(self.bits, (<Bits_u64x4>other).bits)
        elif isinstance(other, Bits_i8x43):
            op_iand(self.bits, (<Bits_i8x43>other).bits)
        else:
            raise TypeError("Unknown input type")
        return self

    def __ixor__(self, other):
        if isinstance(other, Bits_i32x5):
            op_ixor(self.bits, (<Bits_i32x5>other).bits)
        elif isinstance(other, Bits_i64x4):
            op_ixor(self.bits, (<Bits_i64x4>other).bits)
        elif isinstance(other, Bits_u64x4):
            op_ixor(self.bits, (<Bits_u64x4>other).bits)
        elif isinstance(other, Bits_i8x43):
            op_ixor(self.bits, (<Bits_i8x43>other).bits)
        else:
            raise TypeError("Unknown input type")
        return self

cdef class Bits_i64x4:
    cdef _Bits_i64x4 bits
    @staticmethod
    cdef Bits_i64x4 _new(_Bits_i64x4 bits):
        res = <Bits_i64x4>Bits_i64x4.__new__(Bits_i64x4)
        res.bits = bits
        return res

    def __init__(self, bits=0):
        if isinstance(bits, int):
            self.bits = _Bits_i64x4(<int64_t>bits)
        elif isinstance(bits, Bits_i32x5):
            self.bits = convert[_Bits_i64x4,_Bits_i32x5]((<Bits_i32x5>bits).bits)
        elif isinstance(bits, Bits_i64x4):
            self.bits = (<Bits_i64x4>bits).bits
        elif isinstance(bits, Bits_u64x4):
            self.bits = convert[_Bits_i64x4,_Bits_u64x4]((<Bits_u64x4>bits).bits)
        elif isinstance(bits, Bits_i8x43):
            self.bits = convert[_Bits_i64x4,_Bits_i8x43]((<Bits_i8x43>bits).bits)
        else:
            raise TypeError("Unknown input type")

    @staticmethod
    def spec():
        return 4, 64, True

    @staticmethod
    def get_mask(unsigned b1, unsigned b2):
        return Bits_i64x4._new(get_mask[_Bits_i64x4](b1, b2))

    def __richcmp__(self, Bits_i64x4 other, int op):
        if op == Py_EQ:
            return self.bits == other.bits
        if op == Py_NE:
            return self.bits != other.bits
        if op == Py_GT:
            return self.bits > other.bits
        if op == Py_LT:
            return self.bits < other.bits
        if op == Py_GE:
            return self.bits >= other.bits
        if op == Py_LE:
            return self.bits <= other.bits

    def __len__(self):
        return 4

    def __getitem__(self, int i):
        return (<const _Bits_i64x4&>self.bits)[i]

    def __setitem__(self, int i, v):
        self.bits[i] = v

    def __lshift__(self, int i):
        return Bits_i64x4._new(self.bits << i)

    def __rshift__(self, int i):
        return Bits_i64x4._new(self.bits >> i)

    def __index__(self):
        return bits_to_pylong(self.bits)

    def __bool__(self):
        return bits_to_bool(self.bits)

    def bytes(self):
        return bits_to_pybytes(self.bits)

    def __str__(self):
        return bytes(bits_to_str(self.bits, False)).decode()

    def __repr__(self):
        return bytes(bits_to_str(self.bits, True)).decode()

    def __invert__(self):
        return Bits_i64x4._new(op_not(self.bits))

    def __or__(self, other):
        if isinstance(other, Bits_i32x5):
            return Bits_i64x4._new(op_or[_Bits_i64x4,_Bits_i64x4,_Bits_i32x5](
                self.bits, (<Bits_i32x5>other).bits))
        elif isinstance(other, Bits_i64x4):
            return Bits_i64x4._new(op_or[_Bits_i64x4,_Bits_i64x4,_Bits_i64x4](
                self.bits, (<Bits_i64x4>other).bits))
        elif isinstance(other, Bits_u64x4):
            return Bits_i64x4._new(op_or[_Bits_i64x4,_Bits_i64x4,_Bits_u64x4](
                self.bits, (<Bits_u64x4>other).bits))
        elif isinstance(other, Bits_i8x43):
            return Bits_i8x43._new(op_or[_Bits_i8x43,_Bits_i64x4,_Bits_i8x43](
                self.bits, (<Bits_i8x43>other).bits))
        else:
            raise TypeError("Unknown input type")

    def __and__(self, other):
        if isinstance(other, Bits_i32x5):
            return Bits_i64x4._new(op_and[_Bits_i64x4,_Bits_i64x4,_Bits_i32x5](
                self.bits, (<Bits_i32x5>other).bits))
        elif isinstance(other, Bits_i64x4):
            return Bits_i64x4._new(op_and[_Bits_i64x4,_Bits_i64x4,_Bits_i64x4](
                self.bits, (<Bits_i64x4>other).bits))
        elif isinstance(other, Bits_u64x4):
            return Bits_i64x4._new(op_and[_Bits_i64x4,_Bits_i64x4,_Bits_u64x4](
                self.bits, (<Bits_u64x4>other).bits))
        elif isinstance(other, Bits_i8x43):
            return Bits_i8x43._new(op_and[_Bits_i8x43,_Bits_i64x4,_Bits_i8x43](
                self.bits, (<Bits_i8x43>other).bits))
        else:
            raise TypeError("Unknown input type")

    def __xor__(self, other):
        if isinstance(other, Bits_i32x5):
            return Bits_i64x4._new(op_xor[_Bits_i64x4,_Bits_i64x4,_Bits_i32x5](
                self.bits, (<Bits_i32x5>other).bits))
        elif isinstance(other, Bits_i64x4):
            return Bits_i64x4._new(op_xor[_Bits_i64x4,_Bits_i64x4,_Bits_i64x4](
                self.bits, (<Bits_i64x4>other).bits))
        elif isinstance(other, Bits_u64x4):
            return Bits_i64x4._new(op_xor[_Bits_i64x4,_Bits_i64x4,_Bits_u64x4](
                self.bits, (<Bits_u64x4>other).bits))
        elif isinstance(other, Bits_i8x43):
            return Bits_i8x43._new(op_xor[_Bits_i8x43,_Bits_i64x4,_Bits_i8x43](
                self.bits, (<Bits_i8x43>other).bits))
        else:
            raise TypeError("Unknown input type")

    def __ior__(self, other):
        if isinstance(other, Bits_i32x5):
            op_ior(self.bits, (<Bits_i32x5>other).bits)
        elif isinstance(other, Bits_i64x4):
            op_ior(self.bits, (<Bits_i64x4>other).bits)
        elif isinstance(other, Bits_u64x4):
            op_ior(self.bits, (<Bits_u64x4>other).bits)
        elif isinstance(other, Bits_i8x43):
            op_ior(self.bits, (<Bits_i8x43>other).bits)
        else:
            raise TypeError("Unknown input type")
        return self

    def __iand__(self, other):
        if isinstance(other, Bits_i32x5):
            op_iand(self.bits, (<Bits_i32x5>other).bits)
        elif isinstance(other, Bits_i64x4):
            op_iand(self.bits, (<Bits_i64x4>other).bits)
        elif isinstance(other, Bits_u64x4):
            op_iand(self.bits, (<Bits_u64x4>other).bits)
        elif isinstance(other, Bits_i8x43):
            op_iand(self.bits, (<Bits_i8x43>other).bits)
        else:
            raise TypeError("Unknown input type")
        return self

    def __ixor__(self, other):
        if isinstance(other, Bits_i32x5):
            op_ixor(self.bits, (<Bits_i32x5>other).bits)
        elif isinstance(other, Bits_i64x4):
            op_ixor(self.bits, (<Bits_i64x4>other).bits)
        elif isinstance(other, Bits_u64x4):
            op_ixor(self.bits, (<Bits_u64x4>other).bits)
        elif isinstance(other, Bits_i8x43):
            op_ixor(self.bits, (<Bits_i8x43>other).bits)
        else:
            raise TypeError("Unknown input type")
        return self

cdef class Bits_u64x4:
    cdef _Bits_u64x4 bits
    @staticmethod
    cdef Bits_u64x4 _new(_Bits_u64x4 bits):
        res = <Bits_u64x4>Bits_u64x4.__new__(Bits_u64x4)
        res.bits = bits
        return res

    def __init__(self, bits=0):
        if isinstance(bits, int):
            self.bits = _Bits_u64x4(<uint64_t>bits)
        elif isinstance(bits, Bits_i32x5):
            self.bits = convert[_Bits_u64x4,_Bits_i32x5]((<Bits_i32x5>bits).bits)
        elif isinstance(bits, Bits_i64x4):
            self.bits = convert[_Bits_u64x4,_Bits_i64x4]((<Bits_i64x4>bits).bits)
        elif isinstance(bits, Bits_u64x4):
            self.bits = (<Bits_u64x4>bits).bits
        elif isinstance(bits, Bits_i8x43):
            self.bits = convert[_Bits_u64x4,_Bits_i8x43]((<Bits_i8x43>bits).bits)
        else:
            raise TypeError("Unknown input type")

    @staticmethod
    def spec():
        return 4, 64, False

    @staticmethod
    def get_mask(unsigned b1, unsigned b2):
        return Bits_u64x4._new(get_mask[_Bits_u64x4](b1, b2))

    def __richcmp__(self, Bits_u64x4 other, int op):
        if op == Py_EQ:
            return self.bits == other.bits
        if op == Py_NE:
            return self.bits != other.bits
        if op == Py_GT:
            return self.bits > other.bits
        if op == Py_LT:
            return self.bits < other.bits
        if op == Py_GE:
            return self.bits >= other.bits
        if op == Py_LE:
            return self.bits <= other.bits

    def __len__(self):
        return 4

    def __getitem__(self, int i):
        return (<const _Bits_u64x4&>self.bits)[i]

    def __setitem__(self, int i, v):
        self.bits[i] = v

    def __lshift__(self, int i):
        return Bits_u64x4._new(self.bits << i)

    def __rshift__(self, int i):
        return Bits_u64x4._new(self.bits >> i)

    def __index__(self):
        return bits_to_pylong(self.bits)

    def __bool__(self):
        return bits_to_bool(self.bits)

    def bytes(self):
        return bits_to_pybytes(self.bits)

    def __str__(self):
        return bytes(bits_to_str(self.bits, False)).decode()

    def __repr__(self):
        return bytes(bits_to_str(self.bits, True)).decode()

    def __invert__(self):
        return Bits_u64x4._new(op_not(self.bits))

    def __or__(self, other):
        if isinstance(other, Bits_i32x5):
            return Bits_u64x4._new(op_or[_Bits_u64x4,_Bits_u64x4,_Bits_i32x5](
                self.bits, (<Bits_i32x5>other).bits))
        elif isinstance(other, Bits_i64x4):
            return Bits_u64x4._new(op_or[_Bits_u64x4,_Bits_u64x4,_Bits_i64x4](
                self.bits, (<Bits_i64x4>other).bits))
        elif isinstance(other, Bits_u64x4):
            return Bits_u64x4._new(op_or[_Bits_u64x4,_Bits_u64x4,_Bits_u64x4](
                self.bits, (<Bits_u64x4>other).bits))
        elif isinstance(other, Bits_i8x43):
            return Bits_i8x43._new(op_or[_Bits_i8x43,_Bits_u64x4,_Bits_i8x43](
                self.bits, (<Bits_i8x43>other).bits))
        else:
            raise TypeError("Unknown input type")

    def __and__(self, other):
        if isinstance(other, Bits_i32x5):
            return Bits_u64x4._new(op_and[_Bits_u64x4,_Bits_u64x4,_Bits_i32x5](
                self.bits, (<Bits_i32x5>other).bits))
        elif isinstance(other, Bits_i64x4):
            return Bits_u64x4._new(op_and[_Bits_u64x4,_Bits_u64x4,_Bits_i64x4](
                self.bits, (<Bits_i64x4>other).bits))
        elif isinstance(other, Bits_u64x4):
            return Bits_u64x4._new(op_and[_Bits_u64x4,_Bits_u64x4,_Bits_u64x4](
                self.bits, (<Bits_u64x4>other).bits))
        elif isinstance(other, Bits_i8x43):
            return Bits_i8x43._new(op_and[_Bits_i8x43,_Bits_u64x4,_Bits_i8x43](
                self.bits, (<Bits_i8x43>other).bits))
        else:
            raise TypeError("Unknown input type")

    def __xor__(self, other):
        if isinstance(other, Bits_i32x5):
            return Bits_u64x4._new(op_xor[_Bits_u64x4,_Bits_u64x4,_Bits_i32x5](
                self.bits, (<Bits_i32x5>other).bits))
        elif isinstance(other, Bits_i64x4):
            return Bits_u64x4._new(op_xor[_Bits_u64x4,_Bits_u64x4,_Bits_i64x4](
                self.bits, (<Bits_i64x4>other).bits))
        elif isinstance(other, Bits_u64x4):
            return Bits_u64x4._new(op_xor[_Bits_u64x4,_Bits_u64x4,_Bits_u64x4](
                self.bits, (<Bits_u64x4>other).bits))
        elif isinstance(other, Bits_i8x43):
            return Bits_i8x43._new(op_xor[_Bits_i8x43,_Bits_u64x4,_Bits_i8x43](
                self.bits, (<Bits_i8x43>other).bits))
        else:
            raise TypeError("Unknown input type")

    def __ior__(self, other):
        if isinstance(other, Bits_i32x5):
            op_ior(self.bits, (<Bits_i32x5>other).bits)
        elif isinstance(other, Bits_i64x4):
            op_ior(self.bits, (<Bits_i64x4>other).bits)
        elif isinstance(other, Bits_u64x4):
            op_ior(self.bits, (<Bits_u64x4>other).bits)
        elif isinstance(other, Bits_i8x43):
            op_ior(self.bits, (<Bits_i8x43>other).bits)
        else:
            raise TypeError("Unknown input type")
        return self

    def __iand__(self, other):
        if isinstance(other, Bits_i32x5):
            op_iand(self.bits, (<Bits_i32x5>other).bits)
        elif isinstance(other, Bits_i64x4):
            op_iand(self.bits, (<Bits_i64x4>other).bits)
        elif isinstance(other, Bits_u64x4):
            op_iand(self.bits, (<Bits_u64x4>other).bits)
        elif isinstance(other, Bits_i8x43):
            op_iand(self.bits, (<Bits_i8x43>other).bits)
        else:
            raise TypeError("Unknown input type")
        return self

    def __ixor__(self, other):
        if isinstance(other, Bits_i32x5):
            op_ixor(self.bits, (<Bits_i32x5>other).bits)
        elif isinstance(other, Bits_i64x4):
            op_ixor(self.bits, (<Bits_i64x4>other).bits)
        elif isinstance(other, Bits_u64x4):
            op_ixor(self.bits, (<Bits_u64x4>other).bits)
        elif isinstance(other, Bits_i8x43):
            op_ixor(self.bits, (<Bits_i8x43>other).bits)
        else:
            raise TypeError("Unknown input type")
        return self

cdef class Bits_i8x43:
    cdef _Bits_i8x43 bits
    @staticmethod
    cdef Bits_i8x43 _new(_Bits_i8x43 bits):
        res = <Bits_i8x43>Bits_i8x43.__new__(Bits_i8x43)
        res.bits = bits
        return res

    def __init__(self, bits=0):
        if isinstance(bits, int):
            self.bits = _Bits_i8x43(<int>bits)
        elif isinstance(bits, Bits_i32x5):
            self.bits = convert[_Bits_i8x43,_Bits_i32x5]((<Bits_i32x5>bits).bits)
        elif isinstance(bits, Bits_i64x4):
            self.bits = convert[_Bits_i8x43,_Bits_i64x4]((<Bits_i64x4>bits).bits)
        elif isinstance(bits, Bits_u64x4):
            self.bits = convert[_Bits_i8x43,_Bits_u64x4]((<Bits_u64x4>bits).bits)
        elif isinstance(bits, Bits_i8x43):
            self.bits = (<Bits_i8x43>bits).bits
        else:
            raise TypeError("Unknown input type")

    @staticmethod
    def spec():
        return 43, 8, True

    @staticmethod
    def get_mask(unsigned b1, unsigned b2):
        return Bits_i8x43._new(get_mask[_Bits_i8x43](b1, b2))

    def __richcmp__(self, Bits_i8x43 other, int op):
        if op == Py_EQ:
            return self.bits == other.bits
        if op == Py_NE:
            return self.bits != other.bits
        if op == Py_GT:
            return self.bits > other.bits
        if op == Py_LT:
            return self.bits < other.bits
        if op == Py_GE:
            return self.bits >= other.bits
        if op == Py_LE:
            return self.bits <= other.bits

    def __len__(self):
        return 43

    def __getitem__(self, int i):
        return (<const _Bits_i8x43&>self.bits)[i]

    def __setitem__(self, int i, v):
        self.bits[i] = v

    def __lshift__(self, int i):
        return Bits_i8x43._new(self.bits << i)

    def __rshift__(self, int i):
        return Bits_i8x43._new(self.bits >> i)

    def __index__(self):
        return bits_to_pylong(self.bits)

    def __bool__(self):
        return bits_to_bool(self.bits)

    def bytes(self):
        return bits_to_pybytes(self.bits)

    def __str__(self):
        return bytes(bits_to_str(self.bits, False)).decode()

    def __repr__(self):
        return bytes(bits_to_str(self.bits, True)).decode()

    def __invert__(self):
        return Bits_i8x43._new(op_not(self.bits))

    def __or__(self, other):
        if isinstance(other, Bits_i32x5):
            return Bits_i8x43._new(op_or[_Bits_i8x43,_Bits_i8x43,_Bits_i32x5](
                self.bits, (<Bits_i32x5>other).bits))
        elif isinstance(other, Bits_i64x4):
            return Bits_i8x43._new(op_or[_Bits_i8x43,_Bits_i8x43,_Bits_i64x4](
                self.bits, (<Bits_i64x4>other).bits))
        elif isinstance(other, Bits_u64x4):
            return Bits_i8x43._new(op_or[_Bits_i8x43,_Bits_i8x43,_Bits_u64x4](
                self.bits, (<Bits_u64x4>other).bits))
        elif isinstance(other, Bits_i8x43):
            return Bits_i8x43._new(op_or[_Bits_i8x43,_Bits_i8x43,_Bits_i8x43](
                self.bits, (<Bits_i8x43>other).bits))
        else:
            raise TypeError("Unknown input type")

    def __and__(self, other):
        if isinstance(other, Bits_i32x5):
            return Bits_i8x43._new(op_and[_Bits_i8x43,_Bits_i8x43,_Bits_i32x5](
                self.bits, (<Bits_i32x5>other).bits))
        elif isinstance(other, Bits_i64x4):
            return Bits_i8x43._new(op_and[_Bits_i8x43,_Bits_i8x43,_Bits_i64x4](
                self.bits, (<Bits_i64x4>other).bits))
        elif isinstance(other, Bits_u64x4):
            return Bits_i8x43._new(op_and[_Bits_i8x43,_Bits_i8x43,_Bits_u64x4](
                self.bits, (<Bits_u64x4>other).bits))
        elif isinstance(other, Bits_i8x43):
            return Bits_i8x43._new(op_and[_Bits_i8x43,_Bits_i8x43,_Bits_i8x43](
                self.bits, (<Bits_i8x43>other).bits))
        else:
            raise TypeError("Unknown input type")

    def __xor__(self, other):
        if isinstance(other, Bits_i32x5):
            return Bits_i8x43._new(op_xor[_Bits_i8x43,_Bits_i8x43,_Bits_i32x5](
                self.bits, (<Bits_i32x5>other).bits))
        elif isinstance(other, Bits_i64x4):
            return Bits_i8x43._new(op_xor[_Bits_i8x43,_Bits_i8x43,_Bits_i64x4](
                self.bits, (<Bits_i64x4>other).bits))
        elif isinstance(other, Bits_u64x4):
            return Bits_i8x43._new(op_xor[_Bits_i8x43,_Bits_i8x43,_Bits_u64x4](
                self.bits, (<Bits_u64x4>other).bits))
        elif isinstance(other, Bits_i8x43):
            return Bits_i8x43._new(op_xor[_Bits_i8x43,_Bits_i8x43,_Bits_i8x43](
                self.bits, (<Bits_i8x43>other).bits))
        else:
            raise TypeError("Unknown input type")

    def __ior__(self, other):
        if isinstance(other, Bits_i32x5):
            op_ior(self.bits, (<Bits_i32x5>other).bits)
        elif isinstance(other, Bits_i64x4):
            op_ior(self.bits, (<Bits_i64x4>other).bits)
        elif isinstance(other, Bits_u64x4):
            op_ior(self.bits, (<Bits_u64x4>other).bits)
        elif isinstance(other, Bits_i8x43):
            op_ior(self.bits, (<Bits_i8x43>other).bits)
        else:
            raise TypeError("Unknown input type")
        return self

    def __iand__(self, other):
        if isinstance(other, Bits_i32x5):
            op_iand(self.bits, (<Bits_i32x5>other).bits)
        elif isinstance(other, Bits_i64x4):
            op_iand(self.bits, (<Bits_i64x4>other).bits)
        elif isinstance(other, Bits_u64x4):
            op_iand(self.bits, (<Bits_u64x4>other).bits)
        elif isinstance(other, Bits_i8x43):
            op_iand(self.bits, (<Bits_i8x43>other).bits)
        else:
            raise TypeError("Unknown input type")
        return self

    def __ixor__(self, other):
        if isinstance(other, Bits_i32x5):
            op_ixor(self.bits, (<Bits_i32x5>other).bits)
        elif isinstance(other, Bits_i64x4):
            op_ixor(self.bits, (<Bits_i64x4>other).bits)
        elif isinstance(other, Bits_u64x4):
            op_ixor(self.bits, (<Bits_u64x4>other).bits)
        elif isinstance(other, Bits_i8x43):
            op_ixor(self.bits, (<Bits_i8x43>other).bits)
        else:
            raise TypeError("Unknown input type")
        return self

cdef class PyBytesStream:
    cdef utils.pybytes_ostream stm

    def put(self, char c):
        self.stm.put(c)

    def write(self, str s):
        b = <bytes>s.encode()
        self.stm.write(PyBytes_AS_STRING(b), PyBytes_GET_SIZE(b))

    def seek(self, ssize_t p, _dir=None):
        if _dir is None:
            self.stm.seekp(p)
            return
        cdef utils.seekdir dir
        if _dir == 'beg':
            dir = utils.seekdir_beg
        elif _dir == 'end':
            dir = utils.seekdir_end
        elif _dir == 'cur':
            dir = utils.seekdir_cur
        else:
            raise ValueError(f"Invalid seek direction {_dir}")
        self.stm.seekp2(p, dir)

    def flush(self):
        self.stm.flush()

    def get_buf(self):
        return self.stm.get_buf()

    def fail(self):
        return self.stm.fail()

    def clear(self):
        self.stm.clear()

cdef class PyByteArrayStream:
    cdef utils.pybytearray_ostream stm

    def put(self, char c):
        self.stm.put(c)

    def write(self, str s):
        b = <bytes>s.encode()
        self.stm.write(PyBytes_AS_STRING(b), PyBytes_GET_SIZE(b))

    def seek(self, ssize_t p, _dir=None):
        if _dir is None:
            self.stm.seekp(p)
            return
        cdef utils.seekdir dir
        if _dir == 'beg':
            dir = utils.seekdir_beg
        elif _dir == 'end':
            dir = utils.seekdir_end
        elif _dir == 'cur':
            dir = utils.seekdir_cur
        else:
            raise ValueError(f"Invalid seek direction {_dir}")
        self.stm.seekp2(p, dir)

    def flush(self):
        self.stm.flush()

    def get_buf(self):
        buf = <bytearray?>self.stm.get_buf()
        return bytes(buf)

    def fail(self):
        return self.stm.fail()

    def clear(self):
        self.stm.clear()

cdef class IOBuff:
    cdef utils.stringio io

    def write(self, str s):
        self.io.write(s)

    def write_ascii(self, bytes s):
        self.io.write_ascii(s)

    def write_rep_ascii(self, int nrep, bytes s):
        self.io.write_rep_ascii(nrep, s)

    def getvalue(self):
        return self.io.getvalue().rel()

def int_to_chars(int i):
    cdef char buff[5]
    ptr = to_chars(buff, i)
    return PyBytes_FromStringAndSize(buff, ptr - buff)

def int_throw_if(int i):
    return throw_if(i)

def int_throw_if_not(int i):
    return throw_if_not(i)

def test_istream_seek(ssize_t p, _dir=None):
    cdef test_istream stm
    if _dir is None:
        stm.seekg(p)
        return stm.fail()
    cdef utils.seekdir dir
    if _dir == 'beg':
        dir = utils.seekdir_beg
    elif _dir == 'end':
        dir = utils.seekdir_end
    elif _dir == 'cur':
        dir = utils.seekdir_cur
    else:
        raise ValueError(f"Invalid seek direction {_dir}")
    stm.seekg2(p, dir)
    return stm.fail()

def test_istream_ba_seek(ssize_t p, _dir=None):
    cdef test_istream_ba stm
    if _dir is None:
        stm.seekg(p)
        return stm.fail()
    cdef utils.seekdir dir
    if _dir == 'beg':
        dir = utils.seekdir_beg
    elif _dir == 'end':
        dir = utils.seekdir_end
    elif _dir == 'cur':
        dir = utils.seekdir_cur
    else:
        raise ValueError(f"Invalid seek direction {_dir}")
    stm.seekg2(p, dir)
    return stm.fail()

def check_num_arg(func_name, nfound, nmin, nmax):
    py_check_num_arg(func_name, nfound, nmin, nmax)

def yaml_io_print(obj, indent=0):
    if not isinstance(indent, int):
        raise TypeError(f"Unexpected type {type(indent)} for indent")
    if indent < 0:
        raise TypeError("indent cannot be negative")
    cdef utils.stringio io
    _yaml_io_print(io, obj, indent)
    return io.getvalue().rel()
