# cython: language_level=3

# Copyright (c) 2024 - 2025 Yichao Yu <yyc1992@gmail.com>

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
from brassboard_seq.utils cimport PyErr_Format, PyExc_ValueError

from cpython cimport PyObject

cdef extern from "src/event_time.h" namespace "brassboard_seq::event_time":
    # Cython doesn't seem to allow namespace in the object property
    # for the imported extension class
    """
    using _brassboard_seq_event_time_TimeManager = brassboard_seq::event_time::TimeManager;
    using _brassboard_seq_event_time_EventTime = brassboard_seq::event_time::EventTime;
    static inline auto new_time_manager()
    {
        return brassboard_seq::event_time::TimeManager::alloc();
    }
    static inline auto timemanager_new_time_int(auto self, auto prev, auto offset, auto floating, auto cond, auto wait_for)
    {
        return self->new_int(prev, offset, floating, cond, wait_for);
    }
    """
    cppclass EventTimeData:
        int id
        bint floating
        int chain_id
        bint is_static()
        int64_t _get_static()
        int64_t get_static()
        void set_static(int64_t value)
        int64_t get_c_offset()
        void set_c_offset(int64_t value)
        PyObject *get_rt_offset()
        void set_rt_offset(RuntimeValue)

    cppclass TimeManagerStatus:
        int ntimes
        bint finalized

    EventTime new_time_int "timemanager_new_time_int" (TimeManager self, EventTime prev,
                                                       int64_t offset, bint floating,
                                                       object cond,
                                                       EventTime wait_for) except +

    int64_t round_time_f64(double v)
    int64_t round_time_int(v) except +
    RuntimeValue round_time_rt(RuntimeValue) except +

    enum TimeOrder:
        NoOrder
        OrderBefore
        OrderEqual
        OrderAfter

    TimeOrder is_ordered(EventTime t1, EventTime t2) except +

    ctypedef class brassboard_seq._utils.TimeManager [object _brassboard_seq_event_time_TimeManager]:
        cdef shared_ptr[TimeManagerStatus] status
        cdef list event_times
        cdef vector[int64_t] time_values

    TimeManager new_time_manager "new_time_manager"() except +

    ctypedef class brassboard_seq._utils.EventTime [object _brassboard_seq_event_time_EventTime]:
        cdef shared_ptr[TimeManagerStatus] manager_status
        cdef EventTime prev
        cdef EventTime wait_for
        cdef object cond

        cdef EventTimeData data

        cdef vector[int] chain_pos
