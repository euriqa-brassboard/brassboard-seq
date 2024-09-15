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

from cpython cimport PyObject

from libc.stdint cimport *

cdef extern from "src/utils.h" namespace "brassboard_seq":
    T assume[T](T) noexcept nogil
    void assume_not_none(object) noexcept nogil
    void _assume_not_none "brassboard_seq::assume_not_none"(void*) noexcept nogil

    cppclass BacktraceTracker:
        int max_frame
        void record(uintptr_t key)

        cppclass GlobalRestorer:
            pass

    BacktraceTracker.GlobalRestorer set_global_tracker(BacktraceTracker*)

    uintptr_t event_time_key(void*)
    uintptr_t action_key(int)
    uintptr_t assert_key(int)

    int bb_raise(object exc, uintptr_t key) except 0
    int bb_err_format(object exc, uintptr_t key, const char *format, ...) except 0
    PyObject *PyErr_Format(PyObject *exception, char *format, ...) except NULL

    cppclass py_object[T]:
        T *get()
        void reset(T*)

    object pyfloat_from_double(double v)
    list new_list_of_list(int n) except +
    object pynum_add_or_sub(object a, object b, bint issub)
    tuple pytuple_append1(tuple, object) except +
    object pydict_deepcopy(object) except +

# Cython does not declare these in cpython cimport
cdef extern from *:
    PyObject *PyExc_AttributeError
    PyObject *PyExc_KeyError
    PyObject *PyExc_IndexError
    PyObject *PyExc_SyntaxError
    PyObject *PyExc_TypeError
    PyObject *PyExc_ValueError
    PyObject *PyExc_RuntimeError
    PyObject *Py_NotImplemented
    Py_ssize_t PyDict_GET_SIZE(object)
