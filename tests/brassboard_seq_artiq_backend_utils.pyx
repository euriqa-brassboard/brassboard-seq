# cython: language_level=3

from brassboard_seq cimport artiq_backend
from libc.stdint cimport *

cdef class UrukulBus:
    cdef public uint32_t channel
    cdef public uint32_t addr_target
    cdef public uint32_t data_target
    cdef public uint32_t io_update_target
    cdef public uint8_t ref_period_mu

cdef UrukulBus new_urukul_bus(artiq_backend.UrukulBus bus):
    self = <UrukulBus>UrukulBus.__new__(UrukulBus)
    self.channel = bus.channel
    self.addr_target = bus.addr_target
    self.data_target = bus.data_target
    self.io_update_target = bus.io_update_target
    self.ref_period_mu = bus.ref_period_mu
    return self

cdef class DDSChannel:
    cdef public double ftw_per_hz
    cdef public uint32_t bus_id
    cdef public uint8_t chip_select

cdef class TTLChannel:
    cdef public uint32_t target
    cdef public bint iscounter

cdef new_ttl_channel(artiq_backend.TTLChannel *ttl):
    self = <TTLChannel>TTLChannel.__new__(TTLChannel)
    self.target = ttl.target
    self.iscounter = ttl.iscounter
    return self

cdef DDSChannel new_dds_channel(artiq_backend.DDSChannel dds):
    self = <DDSChannel>DDSChannel.__new__(DDSChannel)
    self.ftw_per_hz = dds.ftw_per_hz
    self.bus_id = dds.bus_id
    self.chip_select = dds.chip_select
    return self

class ChannelsInfo:
    pass

def get_channel_info(artiq_backend.ArtiqBackend ab):
    self = ChannelsInfo()
    info = &ab.channels
    self.urukul_busses = [new_urukul_bus(bus) for bus in info.urukul_busses]
    self.ttlchns = [new_ttl_channel(&info.ttlchns[i]) for i in range(info.ttlchns.size())]
    self.ddschns = [new_dds_channel(dds) for dds in info.ddschns]

    self.bus_chn_map = dict(info.bus_chn_map)
    self.ttl_chn_map = dict(info.ttl_chn_map)
    self.dds_param_chn_map = dict(info.dds_param_chn_map)
    return self
