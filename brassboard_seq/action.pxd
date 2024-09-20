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

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.rtval cimport InterpFunction, TagVal
from brassboard_seq.utils cimport py_object

from libcpp.memory cimport unique_ptr

cdef extern from "src/action.h" namespace "brassboard_seq::action":
    cppclass ActionData:
        bint is_pulse;
        bint exact_time;
        bint cond_val;

cdef class Action:
    cdef object value
    cdef object cond
    cdef dict kws
    cdef ActionData data
    cdef int aid

    # Used for processing after the sequence has been programmed only.
    cdef int tid
    cdef int end_tid
    cdef object length
    cdef object prev_val
    cdef object end_val

cdef inline Action new_action(value, cond, bint is_pulse, bint exact_time, dict kws, int aid):
    self = <Action>Action.__new__(Action)
    self.value = value
    self.cond = cond
    self.data.is_pulse = is_pulse
    self.data.exact_time = exact_time
    if kws is not None:
        self.kws = kws
    self.aid = aid
    return self

cdef class RampFunction:
    cdef dict params
    cdef object _eval
    cdef object _spline_segments
    cdef object _fvalue
    cdef unique_ptr[InterpFunction] interp_func

cdef class SeqCubicSpline(RampFunction):
    cdef double order0
    cdef double order1
    cdef double order2
    cdef double order3
    cdef bint compile_mode

cdef ramp_eval(RampFunction self, t, length, oldval)
cdef int ramp_set_compile_params(RampFunction self, length, oldval) except -1
cdef int ramp_set_runtime_params(RampFunction self, unsigned age,
                                 py_object &pyage) except -1
cdef inline ramp_get_spline_segments(RampFunction self, length, oldval):
    if self._spline_segments is None:
        return None
    if type(self) is SeqCubicSpline:
        return ()
    return self._spline_segments(self, length, oldval)
cdef TagVal ramp_interp_eval(RampFunction self, double t) noexcept
