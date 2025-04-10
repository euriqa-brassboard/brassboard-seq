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
    # Cython doesn't seem to allow namespace in the object property
    # for the imported extension class
    """
    using _brassboard_seq_scan_ParamPack = brassboard_seq::scan::ParamPack;
    static inline auto _new_empty_param_pack()
    {
        return brassboard_seq::scan::ParamPack::new_empty();
    }
    """
    ParamPack new_empty_param_pack "_new_empty_param_pack" () except +

    ctypedef class brassboard_seq._utils.ParamPack [object _brassboard_seq_scan_ParamPack, check_size ignore]:
        cdef dict values
        cdef dict visited
        cdef str fieldname
