# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.seq cimport Seq
from brassboard_seq.backend cimport Backend

from libcpp.map cimport map as cppmap
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libc.stdint cimport *

cdef extern from "src/artiq_backend.h" namespace "artiq_backend":
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

    cppclass UrukulBus:
        uint32_t channel
        uint32_t addr_target
        uint32_t data_target
        uint32_t io_update_target
        uint8_t ref_period_mu

    cppclass TTLChannel:
        uint32_t target
        bint iscounter

    cppclass ChannelsInfo:
        vector[UrukulBus] urukul_busses
        vector[TTLChannel] ttlchns
        vector[DDSChannel] ddschns

        cppmap[int,int] bus_chn_map
        cppmap[int,int] ttl_chn_map
        cppmap[pair[int,int],int] dds_chn_map
        cppmap[int,pair[int,ChannelType]] dds_param_chn_map

        int find_bus_id(int bus_channel);
        int add_bus_channel(int bus_channel, uint32_t io_update_target,
                            uint8_t ref_period_mu)
        void add_ttl_channel(int seqchn, uint32_t target, bint iscounter)
        void add_dds_param_channel(int seqchn, uint32_t bus_id, double ftw_per_hz,
                                   uint8_t chip_select, ChannelType param)

    cppclass Relocation:
        int cond_idx
        int time_idx
        int val_idx

    int64_t seq_time_to_mu(long long time)


cdef class ArtiqBackend(Backend):
    # Artiq system object
    cdef sys

    cdef ChannelsInfo channels
    cdef vector[ArtiqAction] all_actions
    cdef vector[pair[void*,bint]] bool_values
    cdef vector[pair[void*,double]] float_values
    cdef vector[Relocation] relocations

    cdef int finalize(self) except -1
