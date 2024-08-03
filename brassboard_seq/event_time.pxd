# cython: language_level=3

from libcpp.memory cimport shared_ptr
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector

from libc.stdint cimport *

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.rtval cimport is_rtval, new_expr2, round_int64_rt, \
  RuntimeValue, ValueType
from brassboard_seq.utils cimport assume_not_none

from cpython cimport PyErr_Format, PyFloat_AS_DOUBLE, PyList_GET_SIZE

# 1ps for internal time unit
cdef extern from "src/event_time.h" namespace "brassboard_seq::event_time":
    long long c_time_scale "brassboard_seq::event_time::time_scale"
    cppclass EventTimeData:
        int id
        bint floating
        int chain_id # ID of the chain this time is part of
        bint has_static
        uint64_t c_offset
        long long _get_static()
        long long get_static()
        void set_static(long long value)

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
        if offset < 0:
            PyErr_Format(ValueError, "Time delay cannot be negative")
        tp.data.has_static = False
        tp.data.c_offset = offset
        tp.data.floating = floating
        tp.cond = cond
        event_times = self.event_times
        cdef int ntimes = status.ntimes
        tp.data.id = ntimes
        assume_not_none(event_times)
        event_times.append(tp)
        status.ntimes = ntimes + 1
        return tp

    cdef inline EventTime new_time_rt(self, EventTime prev, RuntimeValue offset,
                                      cond, EventTime wait_for):
        status = self.status.get()
        if status.finalized:
            PyErr_Format(RuntimeError, "Cannot allocate more time: already finalized")
        tp = <EventTime>EventTime.__new__(EventTime)
        tp.manager_status = self.status
        tp.prev = prev
        tp.wait_for = wait_for
        tp.rt_offset = offset
        tp.data.floating = False
        tp.data.has_static = False
        tp.cond = cond
        event_times = self.event_times
        cdef int ntimes = status.ntimes
        tp.data.id = ntimes
        assume_not_none(event_times)
        event_times.append(tp)
        status.ntimes = ntimes + 1
        return tp

    cdef inline EventTime new_round_time(self, EventTime prev, offset,
                                         cond, EventTime wait_for):
        if is_rtval(offset):
            return self.new_time_rt(prev, round_time_rt(<RuntimeValue>offset),
                                    cond, wait_for)
        else:
            return self.new_time_int(prev, round_time_int(offset),
                                     False, cond, wait_for)

    cdef int finalize(self) except -1
    cdef long long compute_all_times(self, unsigned age) except -1

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
    cdef EventTime prev
    cdef EventTime wait_for
    # If cond is false, this time point is the same as prev
    cdef object cond

    cdef RuntimeValue rt_offset

    cdef EventTimeData data

    # The largest index in each chain that we are no earlier than,
    # In particular for our own chain, this is the position we are in
    cdef vector[int] chain_pos

# All values are in units of `1/time_scale` seconds
cdef inline int set_base_int(EventTime self, EventTime base,
                             long long offset) except -1:
    if not self.data.floating:
        PyErr_Format(ValueError, "Cannot modify non-floating time")
    self.prev = base
    if offset < 0:
        PyErr_Format(ValueError, "Time delay cannot be negative")
    self.data.c_offset = offset
    self.data.floating = False
    return 0

cdef inline int set_base_rt(EventTime self, EventTime base,
                            RuntimeValue offset) except -1:
    if not self.data.floating:
        PyErr_Format(ValueError, "Cannot modify non-floating time")
    self.prev = base
    self.rt_offset = offset
    self.data.floating = False
    return 0

cdef enum TimeOrder:
    NoOrder,
    OrderBefore,
    OrderEqual,
    OrderAfter,
    OrderError

cdef TimeOrder is_ordered(EventTime t1, EventTime t2) except OrderError
