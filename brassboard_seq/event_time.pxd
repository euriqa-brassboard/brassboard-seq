# cython: language_level=3

from libcpp.memory cimport shared_ptr
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.rtval cimport is_rtval, new_expr2, round_int64_rt, \
  RuntimeValue, ValueType
from brassboard_seq.utils cimport assume_not_none

from cpython cimport PyErr_Format, PyFloat_AS_DOUBLE, PyList_GET_SIZE

# 1ps for internal time unit
cdef extern from *:
    """
    static const long long c_time_scale = 1000000000000ll;
    """
    long long c_time_scale

cdef object py_time_scale
cdef RuntimeValue rt_time_scale

cdef inline long long round_time_int(v):
    if type(v) is int:
        return (<long long>v) * c_time_scale
    else:
        vf = (<double>v) * <double>c_time_scale
        return <long long>(vf + 0.5)

cdef inline RuntimeValue round_time_rt(RuntimeValue v):
    return round_int64_rt(new_expr2(ValueType.Mul, v, rt_time_scale))

cdef str _str_time(t)

cdef cppclass TimeManagerStatus:
    int ntimes
    bint finalized

cdef class TimeManager:
    cdef shared_ptr[TimeManagerStatus] status
    cdef list event_times
    cdef vector[long long] time_values

    cdef inline EventTime new_time_int(self, EventTime prev, long long offset,
                                       bint floating, cond, EventTime wait_for):
        status = self.status.get()
        if status.finalized:
            PyErr_Format(RuntimeError, "Cannot allocate more time: already finalized")
        tp = <EventTime>EventTime.__new__(EventTime)
        tp.manager_status = self.status
        tp.prev = prev
        tp.wait_for = wait_for
        tp.c_offset = offset
        tp.floating = floating
        tp.cond = cond
        event_times = self.event_times
        cdef int ntimes = status.ntimes
        tp.id = ntimes
        assume_not_none(event_times)
        event_times.append(tp)
        status.ntimes = ntimes + 1
        return tp

    cdef inline EventTime new_time_rt(self, EventTime prev, RuntimeValue offset,
                                      bint floating, cond, EventTime wait_for):
        status = self.status.get()
        if status.finalized:
            PyErr_Format(RuntimeError, "Cannot allocate more time: already finalized")
        tp = <EventTime>EventTime.__new__(EventTime)
        tp.manager_status = self.status
        tp.prev = prev
        tp.wait_for = wait_for
        tp.rt_offset = offset
        tp.floating = floating
        tp.cond = cond
        event_times = self.event_times
        cdef int ntimes = status.ntimes
        tp.id = ntimes
        assume_not_none(event_times)
        event_times.append(tp)
        status.ntimes = ntimes + 1
        return tp

    cdef inline EventTime new_round_time(self, EventTime prev, offset,
                                         bint floating, cond, EventTime wait_for):
        if is_rtval(offset):
            return self.new_time_rt(prev, round_time_rt(<RuntimeValue>offset),
                                    floating, cond, wait_for)
        else:
            return self.new_time_int(prev, round_time_int(offset),
                                     floating, cond, wait_for)

    cdef int visit_time(self, EventTime t, unordered_set[int] &visited) except -1
    cdef int finalize(self) except -1
    cdef long long compute_all_times(self, long long age) except -1

cdef inline TimeManager new_time_manager():
    self = <TimeManager>TimeManager.__new__(TimeManager)
    self.event_times = []
    status = new TimeManagerStatus()
    status.finalized = False
    status.ntimes = 0
    self.status.reset(status)
    return self

cdef class EventTime:
    cdef shared_ptr[TimeManagerStatus] manager_status
    cdef int id
    cdef EventTime prev
    cdef EventTime wait_for
    # If cond is false, this time point is the same as prev
    cdef object cond

    cdef long long c_offset
    cdef RuntimeValue rt_offset
    cdef bint floating

    cdef int chain_id # ID of the chain this time is part of
    # The largest index in each chain that we are no earlier than,
    # In particular for our own chain, this is the position we are in
    cdef vector[int] chain_pos

    # All values are in units of `1/time_scale` seconds
    cdef int set_base_int(self, EventTime base, long long offset) except -1
    cdef int set_base_rt(self, EventTime base, RuntimeValue offset) except -1
    cdef long long get_value(self, int base_id, long long age,
                             vector[long long] &cache) except -1

cdef enum TimeOrder:
    NoOrder,
    OrderBefore,
    OrderEqual,
    OrderAfter,
    OrderError

cdef TimeOrder is_ordered(EventTime t1, EventTime t2) except OrderError
