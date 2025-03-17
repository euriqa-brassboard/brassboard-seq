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
from brassboard_seq.backend cimport Backend

from libc.stdint cimport *
from libcpp.vector cimport vector
from libcpp.map cimport map as cppmap
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport pair

from cpython cimport PyObject

cdef extern from "src/rfsoc_backend.h" namespace "brassboard_seq::rfsoc_backend":
    struct cubic_spline_t:
        double order0
        double order1
        double order2
        double order3

    enum ToneParam:
        ToneFreq
        TonePhase
        ToneAmp
        ToneFF

    cppclass RFSOCAction:
        bint cond
        bint isramp
        bint sync
        int reloc_id
        int aid
        int tid
        bint is_end
        int64_t seq_time
        double float_value
        bint bool_value
        PyObject *ramp

    cppclass Relocation:
        int cond_idx
        int time_idx
        int val_idx

    cppclass ToneChannel:
        int chn
        vector[RFSOCAction] actions[4]

    cppclass ChannelInfo:
        vector[ToneChannel] channels
        cppmap[int,pair[int,ToneParam]] chn_map
        cppmap[int,int64_t] dds_delay

        int add_tone_channel(int chn) nogil
        void add_seq_channel(int seq_chn, int chn_idx, ToneParam param) nogil
        void set_dds_delay(int dds, int64_t delay) nogil

    cppclass ToneBuffer:
        pass

    cppclass Generator:
        pass

cdef class RFSOCGenerator:
    cdef unique_ptr[Generator] gen

cdef class RFSOCBackend(Backend):
    cdef RFSOCGenerator generator
    cdef ChannelInfo channels
    cdef vector[pair[void*,bint]] bool_values
    cdef vector[pair[void*,double]] float_values
    cdef vector[Relocation] relocations
    cdef bint eval_status
    cdef public bint use_all_channels
    cdef ToneBuffer tone_buffer

    cdef dict rt_dds_delay
