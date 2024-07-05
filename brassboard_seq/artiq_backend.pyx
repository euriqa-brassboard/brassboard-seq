# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.action cimport Action, RampFunction
from brassboard_seq.event_time cimport EventTime
from brassboard_seq.rtval cimport is_rtval
from brassboard_seq.utils cimport set_global_tracker

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

cdef extern from "src/artiq_backend.cpp" namespace "artiq_backend":
    ArtiqConsts artiq_consts

    struct CompileVTable:
        bint (*is_rtval)(object) noexcept
        bint (*is_ramp)(object) noexcept

    void collect_actions(ArtiqBackend ab,
                         CompileVTable vtable, Action, EventTime) except +

cdef inline bint is_ramp(obj) noexcept:
    return isinstance(obj, RampFunction)

cdef inline CompileVTable get_compile_vtable() noexcept nogil:
    cdef CompileVTable vt
    vt.is_rtval = is_rtval
    vt.is_ramp = is_ramp
    return vt

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

cdef get_artiq_device(sys, str name):
    if hasattr(sys, 'registry'):
        # DAX support
        unique = <str?>sys.registry.get_unique_device_key(name)
    else:
        unique = name
    # Do not call the get_device function from DAX since
    # it assumes that the calling object will take ownership of the deivce.
    cls = artiq.language.environment.HasEnvironment
    return cls.get_device(sys, unique)

cdef int add_channel_artiq(ChannelsInfo *self, dev, int idx, tuple path) except -1:
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
        self.add_dds_param_channel(idx, bus_id, <double?>dev.ftw_per_hz,
                                   <int?>dev.chip_select, dds_param_type)
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

cdef int collect_channels(ChannelsInfo *self, str prefix, sys, Seq seq) except -1:
    cdef int idx = -1
    for _path in seq.seqinfo.channel_paths:
        idx += 1
        path = <tuple>_path
        if <str>path[0] != prefix:
            continue
        if len(path) < 2:
            raise_invalid_channel(path)
        add_channel_artiq(self, get_artiq_device(sys, <str>path[1]), idx, path)
    self.dds_chn_map.clear() # Not needed after channel collection
    return 0

@cython.final
cdef class ArtiqBackend:
    def __init__(self, sys):
        self.sys = sys

    cdef int finalize(self) except -1:
        bt_guard = set_global_tracker(&self.seq.seqinfo.bt_tracker)
        collect_channels(&self.channels, self.prefix, self.sys, self.seq)
        collect_actions(self, get_compile_vtable(), None, None)
