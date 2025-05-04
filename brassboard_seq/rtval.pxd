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

cdef extern from "src/rtval.h" namespace "brassboard_seq::rtval":
    # Cython doesn't seem to allow namespace in the object property
    # for the imported extension class
    """
    using _brassboard_seq_rtval_RuntimeValue = brassboard_seq::rtval::RuntimeValue;
    using _brassboard_seq_rtval_ExternCallback = brassboard_seq::rtval::ExternCallback;
    """
    cppclass TagVal:
        pass

    bint is_rtval(object)

    RuntimeValue new_extern(ExternCallback cb, ty) except +
    RuntimeValue new_extern_age(ExternCallback cb, ty) except +

    ctypedef class brassboard_seq.rtval.RuntimeValue [object _brassboard_seq_rtval_RuntimeValue, check_size ignore]:
        pass

    ctypedef class brassboard_seq.rtval.ExternCallback [object _brassboard_seq_rtval_ExternCallback, check_size ignore]:
        cdef void *fptr
