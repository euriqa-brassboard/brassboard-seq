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

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.rtval cimport new_const, new_extern_age, ExternCallback, TagVal
from brassboard_seq.utils cimport PyExc_RuntimeError, PyExc_TypeError

cimport cython

cdef object py_time_scale = c_time_scale
cdef RuntimeValue rt_time_scale = new_const(py_time_scale, <RuntimeValue>None)

def time_scale():
    return py_time_scale

cdef extern from "src/event_time.cpp" namespace "brassboard_seq::event_time":
    str str_time(long long) except +
    void timemanager_finalize(TimeManager self, EventTime) except +
    long long timemanager_compute_all_times(TimeManager self, unsigned age,
                                            py_object &pyage, EventTime) except +
    TagVal timediff_eval(EventTimeDiff self, unsigned age) except +
    void update_event_time_gc_callback(object type, EventTime)

update_event_time_gc_callback(EventTime, None)

@cython.auto_pickle(False)
@cython.final
cdef class TimeManager:
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "TimeManager cannot be created directly")

    cdef int finalize(self) except -1:
        timemanager_finalize(self, None)

    cdef long long compute_all_times(self, unsigned age, py_object &pyage) except -1:
        return timemanager_compute_all_times(self, age, pyage, None)

@cython.auto_pickle(False)
@cython.c_api_binop_methods(True)
@cython.final
cdef class EventTime:
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "EventTime cannot be created directly")

    def __sub__(EventTime self, EventTime other):
        if self.manager_status.get() != other.manager_status.get():
            PyErr_Format(PyExc_ValueError,
                         "Cannot take the difference between unrelated times")
        diff = <EventTimeDiff>EventTimeDiff.__new__(EventTimeDiff)
        diff.t1 = self
        diff.t2 = other
        diff.in_eval = False
        diff.fptr = <void*><TagVal(*)(EventTimeDiff, unsigned)>timediff_eval
        return new_extern_age(diff, float)

    def __str__(self):
        if self.data.floating:
            return '<floating>'
        if self.data.is_static():
            return str_time(self.data._get_static())
        prev = self.prev
        p_rt_offset = self.data.get_rt_offset()
        if p_rt_offset == NULL:
            offset = str_time(<long long>self.data.get_c_offset())
        else:
            offset = str(<RuntimeValue>p_rt_offset)
        cond = self.cond
        if prev is None:
            assert cond is True
            return offset
        wait_for = self.wait_for
        if wait_for is None:
            if cond is True:
                return f'T[{prev.data.id}] + {offset}'
            return f'T[{prev.data.id}] + ({offset}; if {cond})'
        if cond is True:
            return f'T[{prev.data.id}]; wait_for(T[{wait_for.data.id}] + {offset})'
        return f'T[{prev.data.id}]; wait_for(T[{wait_for.data.id}] + {offset}; if {cond})'

    def __repr__(self):
        return str(self)


@cython.internal
@cython.final
cdef class EventTimeDiff(ExternCallback):
    cdef EventTime t1
    cdef EventTime t2
    cdef bint in_eval

    def __str__(self):
        return f'(T[{self.t1.data.id}] - T[{self.t2.data.id}])'
