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

from libcpp.map cimport map as cppmap
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libc.stdint cimport *

cdef extern from "src/artiq_backend.h" namespace "brassboard_seq::artiq_backend":
    # Cython doesn't seem to allow namespace in the object property
    # for the imported extension class
    """
    using _brassboard_seq_artiq_backend_ArtiqBackend = brassboard_seq::artiq_backend::ArtiqBackend;
    """
    enum ChannelType:
        DDSFreq
        DDSAmp
        DDSPhase
        TTLOut
        CounterEnable

    cppclass ArtiqAction:
        ChannelType type
        bint cond
        bint exact_time
        int chn_idx
        int tid
        int64_t time_mu
        uint32_t value
        int aid
        int reloc_id

    cppclass DDSChannel:
        double ftw_per_hz
        uint32_t bus_id
        uint8_t chip_select
        int64_t delay

    cppclass UrukulBus:
        uint32_t channel
        uint32_t addr_target
        uint32_t data_target
        uint32_t io_update_target
        uint8_t ref_period_mu

    cppclass TTLChannel:
        uint32_t target
        bint iscounter
        int64_t delay

    cppclass StartTrigger:
        uint32_t target
        uint16_t min_time_mu
        bint raising_edge
        int64_t time_mu

    cppclass ChannelsInfo:
        vector[UrukulBus] urukul_busses
        vector[TTLChannel] ttlchns
        vector[DDSChannel] ddschns

        cppmap[int,int] bus_chn_map
        cppmap[int,int] ttl_chn_map
        cppmap[pair[int,int],int] dds_chn_map
        cppmap[int,pair[int,ChannelType]] dds_param_chn_map

    cppclass Relocation:
        int cond_idx
        int time_idx
        int val_idx

    ctypedef class brassboard_seq.artiq_backend.ArtiqBackend [object _brassboard_seq_artiq_backend_ArtiqBackend, check_size ignore]:
        pass
