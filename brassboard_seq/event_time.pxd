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

from libcpp.vector cimport vector

from libc.stdint cimport *

# Do not use relative import since it messes up cython file name tracking

cdef extern from "src/event_time.h" namespace "brassboard_seq::event_time":
    # Cython doesn't seem to allow namespace in the object property
    # for the imported extension class
    """
    using _brassboard_seq_event_time_TimeManager = brassboard_seq::event_time::TimeManager;
    using _brassboard_seq_event_time_EventTime = brassboard_seq::event_time::EventTime;
    """
    cppclass EventTimeData:
        int id
        int64_t get_static()

    int64_t round_time_int(v) except +

    enum TimeOrder:
        NoOrder
        OrderBefore
        OrderEqual
        OrderAfter

    TimeOrder is_ordered(EventTime t1, EventTime t2) except +

    ctypedef class brassboard_seq._utils.TimeManager [object _brassboard_seq_event_time_TimeManager]:
        cdef vector[int64_t] time_values

    ctypedef class brassboard_seq.event_time.EventTime [object _brassboard_seq_event_time_EventTime]:
        cdef EventTimeData data

        cdef vector[int] chain_pos
