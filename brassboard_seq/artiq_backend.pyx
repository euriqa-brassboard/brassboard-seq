# cython: language_level=3

cimport cython
from cpython cimport PyErr_Format, PyObject

# Declare these as cdef so that they are hidden from python
# and can be accessed more efficiently from this module.
cdef artiq, ad9910, edge_counter, spi2, ttl, urukul

import artiq.language.environment
from artiq.coredevice import ad9910, edge_counter, spi2, ttl, urukul

cdef DevAD9910 = ad9910.AD9910
cdef DevEdgeCounter = edge_counter.EdgeCounter
cdef DevTTLOut = ttl.TTLOut

cdef extern from "src/artiq_backend.cpp" namespace "artiq_backend" nogil:
    ArtiqConsts artiq_consts

artiq_consts.COUNTER_ENABLE = <int?>edge_counter.CONFIG_COUNT_RISING | <int?>edge_counter.CONFIG_RESET_TO_ZERO
artiq_consts.COUNTER_DISABLE = <int?>edge_counter.CONFIG_SEND_COUNT_EVENT
artiq_consts._AD9910_REG_PROFILE0 = <int?>ad9910._AD9910_REG_PROFILE0
artiq_consts.URUKUL_CONFIG = <int?>urukul.SPI_CONFIG
artiq_consts.URUKUL_CONFIG_END = <int?>urukul.SPI_CONFIG | <int?>spi2.SPI_END
artiq_consts.URUKUL_SPIT_DDS_WR = <int?>urukul.SPIT_DDS_WR
artiq_consts.SPI_DATA_ADDR = <int?>spi2.SPI_DATA_ADDR
artiq_consts.SPI_CONFIG_ADDR = <int?>spi2.SPI_CONFIG_ADDR

cdef PyObject *raise_invalid_channel(tuple path) except NULL:
    name = '/'.join(path)
    return PyErr_Format(ValueError, 'Invalid channel name %U', <PyObject*>name)

@cython.final
cdef class ChannelsInfo:
    def __init__(self):
        PyErr_Format(TypeError, "ChannelInfo cannot be created directly")

    cdef int add_bus_channel(self, int bus_channel, int io_update_target,
                             int ref_period_mu) except -1:
        cdef UrukulBus bus_info
        bus_info.channel = bus_channel
        bus_info.addr_target = (bus_channel << 8) | artiq_consts.SPI_CONFIG_ADDR
        bus_info.data_target = (bus_channel << 8) | artiq_consts.SPI_DATA_ADDR
        # Here we assume that the CPLD (and it's io_update channel)
        # and the SPI bus has a one-to-one mapping.
        # This means that each DDS with the same bus shares
        # the same io_update channel and can only be programmed one at a time.
        bus_info.io_update_target = io_update_target
        bus_info.ref_period_mu = ref_period_mu
        cdef int bus_id = self.urukul_busses.size()
        self.urukul_busses.push_back(bus_info)
        self.bus_chn_map[bus_channel] = bus_id
        return bus_id

    cdef int add_ttl_channel(self, int seqchn, int target, bint iscounter) except -1:
        assert self.ttl_chn_map.count(seqchn) == 0
        cdef TTLChannel ttl_chn
        ttl_chn.target = target
        ttl_chn.iscounter = iscounter
        cdef int ttl_id = self.ttlchns.size()
        self.ttlchns.push_back(ttl_chn)
        self.ttl_chn_map[seqchn] = ttl_id
        return ttl_id

    cdef int get_dds_channel_id(self, int bus_id, double ftw_per_hz,
                                uint8_t chip_select) except -1:
        cdef pair[int,int] key = pair[int,int](bus_id, chip_select)
        if self.dds_chn_map.count(key) != 0:
            return self.dds_chn_map[key]
        cdef DDSChannel dds_chn
        dds_chn.ftw_per_hz = ftw_per_hz
        dds_chn.bus_id = bus_id
        dds_chn.chip_select = chip_select
        cdef int dds_id = self.ddschns.size()
        self.ddschns.push_back(dds_chn)
        self.dds_chn_map[key] = dds_id
        return dds_id

    cdef int add_dds_param_channel(self, int seqchn, int dds_id,
                                   ChannelType param) except -1:
        assert self.dds_param_chn_map.count(seqchn) == 0
        self.dds_param_chn_map[seqchn] = pair[int,ChannelType](dds_id, param)
        return 0

    cdef get_device(self, str name):
        if hasattr(self.sys, 'registry'):
            # DAX support
            unique = <str?>self.sys.registry.get_unique_device_key(name)
        else:
            unique = name
        # Do not call the get_device function from DAX since
        # it assumes that the calling object will take ownership of the deivce.
        cls = artiq.language.environment.HasEnvironment
        return cls.get_device(self.sys, unique)

    cdef int add_channel_artiq(self, int idx, tuple path) except -1:
        dev = self.get_device(<str>path[1])
        cdef ChannelType dds_param_type
        if isinstance(dev, DevAD9910):
            if len(path) != 3:
                raise_invalid_channel(path)
            path2 = <str>path[2]
            if path2 == 'sw':
                # Note that we currently do not treat this switch ttl channel
                # differently from any other ttl channels.
                # We may consider maintaining a relation between this ttl channel
                # and the urukul channel to make sure we don't reorder
                # any operations between the two.
                self.add_ttl_channel(idx, <int?>dev.sw.target_o, False)
                return 0
            elif path2 == 'freq':
                dds_param_type = DDSFreq
            elif path2 == 'amp':
                dds_param_type = DDSAmp
            elif path2 == 'phase':
                dds_param_type = DDSPhase
            else:
                # Make the C compiler happy since it doesn't know
                # that `raise_invalid_channel` doesn't return
                dds_param_type = DDSPhase
                raise_invalid_channel(path)
            bus = dev.bus
            bus_channel = <int?>bus.channel
            bus_id = self.find_bus_id(bus_channel)
            if bus_id == -1:
                # Here we assume that the CPLD (and it's io_update channel)
                # and the SPI bus has a one-to-one mapping.
                # This means that each DDS with the same bus shares
                # the same io_update channel and can only be programmed one at a time.
                io_update_target = <int?>dev.cpld.io_update.target_o
                bus_id = self.add_bus_channel(bus_channel, io_update_target,
                                              <int?>bus.ref_period_mu)
            dds_id = self.get_dds_channel_id(bus_id, <double?>dev.ftw_per_hz,
                                             <int?>dev.chip_select)
            self.add_dds_param_channel(idx, dds_id, dds_param_type)
        elif isinstance(dev, DevTTLOut):
            if len(path) > 2:
                raise_invalid_channel(path)
            self.add_ttl_channel(idx, <int?>dev.target_o, False)
        elif isinstance(dev, DevEdgeCounter):
            if len(path) > 2:
                raise_invalid_channel(path)
            self.add_ttl_channel(idx, (<int?>dev.channel) << 8, True)
        else:
            devstr = str(dev)
            PyErr_Format(ValueError, 'Unsupported device: %U', <PyObject*>devstr)
        return 0

    cdef int collect_channels(self, Seq seq) except -1:
        cdef int idx = -1
        for _path in seq.seqinfo.channel_paths:
            idx += 1
            path = <tuple>_path
            if <str>path[0] != 'artiq':
                continue
            if len(path) < 2:
                raise_invalid_channel(path)
            self.add_channel_artiq(idx, path)
        self.dds_chn_map.clear() # Not needed after channel collection
        return 0

cdef ChannelsInfo new_channels_info(sys):
    self = <ChannelsInfo>ChannelsInfo.__new__(ChannelsInfo)
    self.sys = sys
    return self

@cython.final
cdef class ArtiqBackend:
    def __init__(self):
        PyErr_Format(TypeError, "ArtiqBackend cannot be created directly")

    cdef int process_seq(self) except -1:
        self.channels.collect_channels(self.seq)

cpdef ArtiqBackend new_artiq_backend(sys, Seq seq):
    self = <ArtiqBackend>ArtiqBackend.__new__(ArtiqBackend)
    self.seq = seq
    self.channels = new_channels_info(sys)
    return self
