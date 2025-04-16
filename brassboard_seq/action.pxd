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
from brassboard_seq.rtval cimport InterpFunction, TagVal

from libcpp.memory cimport unique_ptr

from cpython cimport PyObject

cdef extern from "src/action.h" namespace "brassboard_seq::action":
    cppclass Action:
        bint is_pulse
        bint exact_time
        bint cond_val
        int aid
        int tid
        int end_tid

        str py_str() except +

    cppclass ActionAllocator:
        pass

cdef class _RampFunctionBase:
    cdef eval_end(self, length, oldval)
    # Currently this function is also used to pass the runtime length and oldval
    # info to the ramp function to be used in subsequent runtime_eval calls.
    # This may be moved into a different function if/when we have a caller
    # that only need one of the effects of this function.
    # Also note that this API mutates the object and currently means
    # we cannot compute multiple ramps concurrently.
    cdef spline_segments(self, double length, double oldval)
    cdef int set_runtime_params(self, unsigned age) except -1
    cdef TagVal runtime_eval(self, double t) noexcept

cdef class RampFunction(_RampFunctionBase):
    cdef object _eval
    cdef object _spline_segments
    cdef object _fvalue
    cdef unique_ptr[InterpFunction] interp_func

cdef class SeqCubicSpline(_RampFunctionBase):
    cdef readonly object order0
    cdef readonly object order1
    cdef readonly object order2
    cdef readonly object order3
    cdef double f_order0
    cdef double f_order1
    cdef double f_order2
    cdef double f_order3
    cdef double f_inv_length
