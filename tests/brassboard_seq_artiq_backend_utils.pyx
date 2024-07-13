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

cdef class ArtiqAction:
    cdef public str type
    cdef public bint cond
    cdef public bint exact_time
    cdef public int chn_idx
    cdef public int tid
    cdef public int64_t time_mu
    cdef public uint32_t value
    cdef public int aid
    cdef public int reloc_id

channel_type_names = {
    artiq_backend.DDSFreq: 'ddsfreq',
    artiq_backend.DDSAmp: 'ddsamp',
    artiq_backend.DDSPhase: 'ddsphase',
    artiq_backend.TTLOut: 'ttl',
    artiq_backend.CounterEnable: 'counter',
}

cdef ArtiqAction new_artiq_action(artiq_backend.ArtiqAction c_action):
    py_action = <ArtiqAction>ArtiqAction.__new__(ArtiqAction)
    py_action.type = channel_type_names[c_action.type]
    py_action.cond = c_action.cond
    py_action.exact_time = c_action.exact_time
    py_action.chn_idx = c_action.chn_idx
    py_action.tid = c_action.tid
    py_action.time_mu = c_action.time_mu
    py_action.value = c_action.value
    py_action.aid = c_action.aid
    py_action.reloc_id = c_action.reloc_id
    return py_action

cdef class Relocation:
    cdef public int cond_idx
    cdef public int time_idx
    cdef public int val_idx

cdef Relocation new_relocation(artiq_backend.Relocation c_reloc):
    py_reloc = <Relocation>Relocation.__new__(Relocation)
    py_reloc.cond_idx = c_reloc.cond_idx
    py_reloc.time_idx = c_reloc.time_idx
    py_reloc.val_idx = c_reloc.val_idx
    return py_reloc

class CompiledInfo:
    pass

def get_compiled_info(artiq_backend.ArtiqBackend ab):
    self = CompiledInfo()
    self.all_actions = [new_artiq_action(ab.all_actions[i])
                            for i in range(ab.all_actions.size())]
    self.bool_values = [<object>p.first for p in ab.bool_values]
    self.float_values = [<object>p.first for p in ab.float_values]
    self.relocations = [new_relocation(action) for action in ab.relocations]
    return self

cdef class StartTrigger:
    cdef public uint32_t target
    cdef public uint16_t min_time_mu
    cdef public bint raising_edge
    cdef public int64_t time_mu

cdef StartTrigger new_start_trigger(artiq_backend.StartTrigger c_trigger):
    py_trigger = <StartTrigger>StartTrigger.__new__(StartTrigger)
    py_trigger.target = c_trigger.target
    py_trigger.min_time_mu = c_trigger.min_time_mu
    py_trigger.raising_edge = c_trigger.raising_edge
    py_trigger.time_mu = c_trigger.time_mu
    return py_trigger

def get_start_trigger(artiq_backend.ArtiqBackend ab):
    return [new_start_trigger(trigger) for trigger in ab.start_triggers]

def add_start_trigger(artiq_backend.ArtiqBackend ab, uint32_t tgt, long long time,
                      int min_time, bint raising_edge):
    ab.add_start_trigger_ttl(tgt, time, min_time, raising_edge)

def get_start_trigger(artiq_backend.ArtiqBackend ab):
    return [new_start_trigger(trigger) for trigger in ab.start_triggers]
