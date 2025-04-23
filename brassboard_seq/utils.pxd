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

from cpython cimport PyObject

from libc.stdint cimport *

cdef extern from "<ostream>" namespace "std":
    ctypedef struct seekdir "std::ios_base::seekdir":
        pass
    cdef seekdir seekdir_beg "std::ios_base::beg"
    cdef seekdir seekdir_end "std::ios_base::end"
    cdef seekdir seekdir_cur "std::ios_base::cur"
    cdef cppclass ostream:
        ostream &operator<<[T](const T&)
        ostream &put(char)
        ostream &write(const char*, size_t)
        ostream &seekp(ssize_t)
        ostream &seekp2 "seekp"(ssize_t, seekdir)
        ostream &flush()
        bint fail() const
        void clear()

cdef extern from "src/utils.h" namespace "brassboard_seq::py":
    cppclass ref "brassboard_seq::py::ref<>":
        object rel "rel<PyObject>" ()

    cppclass stringio:
        void write(str)
        void write_str(object) except +
        void write_ascii(const char*)
        void write_rep_ascii(int, const char*)
        ref getvalue() except +

    ref new_int[T](T v) except +
    ref new_float(double v) except +

cdef extern from * namespace "brassboard_seq":
    void init() except +
    PyObject *PyErr_Format(PyObject *exception, char *format, ...) except NULL

    cppclass pybytes_ostream(ostream):
        object get_buf() except +

    cppclass pybytearray_ostream(ostream):
        object get_buf() except +

# Cython does not declare these in cpython cimport
cdef extern from *:
    PyObject *Py_None
    PyObject *Py_True
    PyObject *Py_False
    PyObject *PyExc_AttributeError
    PyObject *PyExc_KeyError
    PyObject *PyExc_IndexError
    PyObject *PyExc_SyntaxError
    PyObject *PyExc_TypeError
    PyObject *PyExc_ValueError
    PyObject *PyExc_RuntimeError
    PyObject *Py_NotImplemented
    object PyObject_Vectorcall(PyObject*, PyObject**, size_t, PyObject*)
