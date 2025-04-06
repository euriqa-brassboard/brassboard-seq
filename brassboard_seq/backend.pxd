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
from brassboard_seq.seq cimport Seq

cdef extern from "src/backend.h" namespace "brassboard_seq::backend":
    cppclass CompiledSeq:
        pass

cdef class Backend:
    cdef Seq seq
    cdef str prefix

    cdef int finalize(self, CompiledSeq&) except -1
    cdef int runtime_finalize(self, CompiledSeq&, unsigned age) except -1

cdef class SeqCompiler:
    cdef readonly Seq seq
    cdef CompiledSeq cseq
    cdef dict backends
