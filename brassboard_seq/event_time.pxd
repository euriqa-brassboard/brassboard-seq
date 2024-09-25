# cython: language_level=3

# Copyright (c) 2024 - 2024 Yichao Yu <yyc1992@gmail.com>

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3.0 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not,
# see <http://www.gnu.org/licenses/>.

from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from libc.stdint cimport *

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.rtval cimport is_rtval, RuntimeValue
from brassboard_seq.utils cimport PyErr_Format, PyExc_ValueError, py_object

from cpython cimport PyObject

# 1ps for internal time unit
cdef extern from "src/event_time.h" namespace "brassboard_seq::event_time":
    long long c_time_scale "brassboard_seq::event_time::time_scale"
    cppclass EventTimeData:
        int id
        bint floating
        int chain_id # ID of the chain this time is part of
        bint is_static()
        long long _get_static()
        long long get_static()
        void set_static(long long value)
        long long get_c_offset()
        void set_c_offset(long long value)
        PyObject *get_rt_offset()
        void set_rt_offset(RuntimeValue)

    cppclass TimeManagerStatus:
        int ntimes
        bint finalized

    EventTime _new_time_int(TimeManager self, object EventTimeType,
                            EventTime prev, long long offset,
                            bint floating, object cond, EventTime wait_for) except +
    EventTime _new_time_rt(TimeManager self, object EventTimeType, EventTime prev,
                           RuntimeValue offset, object cond, EventTime wait_for) except +

    long long round_time_f64(double v)
    long long round_time_int(v) except +
    RuntimeValue round_time_rt(object, RuntimeValue, RuntimeValue) except +

cdef object py_time_scale
cdef RuntimeValue rt_time_scale

cdef class TimeManager:
    cdef shared_ptr[TimeManagerStatus] status
    cdef list event_times
    cdef vector[long long] time_values

    cdef inline EventTime new_time_int(self, EventTime prev, long long offset,
                                       bint floating, cond, EventTime wait_for):
        return _new_time_int(self, EventTime, prev, offset, floating, cond, wait_for)

    cdef inline EventTime new_time_rt(self, EventTime prev, RuntimeValue offset,
                                      cond, EventTime wait_for):
        return _new_time_rt(self, EventTime, prev, offset, cond, wait_for)

    cdef inline EventTime new_round_time(self, EventTime prev, offset,
                                         cond, EventTime wait_for):
        if is_rtval(offset):
            return _new_time_rt(self, EventTime, prev,
                                round_time_rt(RuntimeValue, <RuntimeValue>offset,
                                              rt_time_scale), cond, wait_for)
        else:
            return _new_time_int(self, EventTime, prev, round_time_int(offset),
                                 False, cond, wait_for)

    cdef int finalize(self) except -1
    cdef long long compute_all_times(self, unsigned age, py_object &pyage) except -1

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

    cdef EventTimeData data

    # The largest index in each chain that we are no earlier than,
    # In particular for our own chain, this is the position we are in
    cdef vector[int] chain_pos

# All values are in units of `1/time_scale` seconds
cdef inline int set_base_int(EventTime self, EventTime base,
                             long long offset) except -1:
    if not self.data.floating:
        PyErr_Format(PyExc_ValueError, "Cannot modify non-floating time")
    self.prev = base
    if offset < 0:
        PyErr_Format(PyExc_ValueError, "Time delay cannot be negative")
    self.data.set_c_offset(offset)
    self.data.floating = False
    return 0

cdef inline int set_base_rt(EventTime self, EventTime base,
                            RuntimeValue offset) except -1:
    if not self.data.floating:
        PyErr_Format(PyExc_ValueError, "Cannot modify non-floating time")
    self.prev = base
    self.data.set_rt_offset(offset)
    self.data.floating = False
    return 0

cdef enum TimeOrder:
    NoOrder,
    OrderBefore,
    OrderEqual,
    OrderAfter,
    OrderError

cdef TimeOrder is_ordered(EventTime t1, EventTime t2) except OrderError
