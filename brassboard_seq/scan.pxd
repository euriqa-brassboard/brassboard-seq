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

cdef extern from "src/scan.h" namespace "brassboard_seq::scan":
    ParamPack new_param_pack(object, dict value, dict visited,
                             str fieldname, ParamPack) except +

cdef class ParamPack:
    cdef dict values
    cdef dict visited
    cdef str fieldname
    cdef void *vectorcall_ptr
