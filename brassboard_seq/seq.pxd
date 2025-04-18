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
from brassboard_seq.event_time cimport TimeManager

cdef extern from "src/seq.h" namespace "brassboard_seq::seq":
    # Cython doesn't seem to allow namespace in the object property
    # for the imported extension class
    """
    using _brassboard_seq_seq_SeqInfo = brassboard_seq::seq::SeqInfo;
    using _brassboard_seq_seq_Seq = brassboard_seq::seq::Seq;
    """
    ctypedef class brassboard_seq._utils.SeqInfo [object _brassboard_seq_seq_SeqInfo]:
        cdef TimeManager time_mgr
        cdef list channel_paths

    ctypedef class brassboard_seq._utils.Seq [object _brassboard_seq_seq_Seq]:
        cdef SeqInfo seqinfo
