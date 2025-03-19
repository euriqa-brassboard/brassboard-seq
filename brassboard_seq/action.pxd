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

from cpython cimport PyObject

cdef extern from "src/action.h" namespace "brassboard_seq::action":
    cppclass Action:
        Action(object, object, bint, bint, py_object&, int)
        py_object value
        py_object cond
        py_object kws
        bint is_pulse
        bint exact_time
        bint cond_val
        int aid
        int tid
        int end_tid
        PyObject *length
        py_object end_val

    cppclass ActionAllocator:
        pass

cdef str action_str(Action*)

cdef class RampFunction:
    cdef object _eval
    cdef object _spline_segments
    cdef object _fvalue
    cdef unique_ptr[InterpFunction] interp_func

    cdef eval(self, t, length, oldval)
    cdef spline_segments(self, double length, double oldval)
    cdef int set_compile_params(self, length, oldval) except -1
    cdef int set_runtime_params(self, unsigned age, py_object &pyage) except -1
    cdef TagVal runtime_eval(self, double t) noexcept

cdef class SeqCubicSpline(RampFunction):
    # _eval -> length
    # _spline_segments -> order0
    # _fvalue -> order1
    cdef readonly object order2
    cdef readonly object order3
    cdef double f_order0
    cdef double f_order1
    cdef double f_order2
    cdef double f_order3
    cdef double f_inv_length
