# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.seq cimport Seq

from libcpp.map cimport map as cppmap
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libc.stdint cimport *

cdef extern from "src/artiq_backend.h" namespace "artiq_backend" nogil:
    cppclass ArtiqConsts:
        int COUNTER_ENABLE
        int COUNTER_DISABLE
        int _AD9910_REG_PROFILE0
        int URUKUL_CONFIG
        int URUKUL_CONFIG_END
        int URUKUL_SPIT_DDS_WR
        int SPI_CONFIG_ADDR
        int SPI_DATA_ADDR

    enum ChannelType:
        DDSFreq
        DDSAmp
        DDSPhase
        TTLOut
        CounterEnable

    cppclass UrukulBus:
        uint32_t channel
        uint32_t addr_target
        uint32_t data_target
        uint32_t io_update_target
        uint8_t ref_period_mu

    cppclass DDSChannel:
        double ftw_per_hz
        uint32_t bus_id
        uint8_t chip_select

    cppclass TTLChannel:
        uint32_t target
        bint iscounter


cdef class ChannelsInfo:
    # Artiq objects
    cdef sys

    # Sequence info
    cdef vector[UrukulBus] urukul_busses
    cdef vector[TTLChannel] ttlchns
    cdef vector[DDSChannel] ddschns

    # From bus channel to urukul bus index
    cdef cppmap[int,int] bus_chn_map
    # From sequence channel id to ttl channel index
    cdef cppmap[int,int] ttl_chn_map
    # From (bus_id, chip select) to dds channel index
    cdef cppmap[pair[int,int],int] dds_chn_map
    # From sequence channel id to dds channel index + channel type
    cdef cppmap[int,pair[int,ChannelType]] dds_param_chn_map

    cdef inline int find_bus_id(self, int bus_channel) noexcept:
        if self.bus_chn_map.count(bus_channel):
            return self.bus_chn_map[bus_channel]
        return -1
    cdef int add_bus_channel(self, int bus_channel, int io_update_target,
                             int ref_period_mu) except -1
    cdef int add_ttl_channel(self, int seqchn, int target, bint iscounter) except -1
    cdef int get_dds_channel_id(self, int bus_id, double ftw_per_hz,
                                uint8_t chip_select) except -1
    cdef int add_dds_param_channel(self, int seqchn, int dds_id,
                                   ChannelType param) except -1

    cdef get_device(self, str name)

    cdef int add_channel_artiq(self, int idx, tuple path) except -1
    cdef int collect_channels(self, Seq seq) except -1

cdef class ArtiqBackend:
    cdef Seq seq
    cdef ChannelsInfo channels

    cdef int process_seq(self) except -1

cpdef ArtiqBackend new_artiq_backend(sys, Seq seq)
