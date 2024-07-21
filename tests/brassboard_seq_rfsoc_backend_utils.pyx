# cython: language_level=3

from brassboard_seq cimport rfsoc_backend

from libc.stdint cimport *

class RFSOCAction:
    pass

cdef new_rfsoc_action(rfsoc_backend.RFSOCAction *action):
    self = RFSOCAction()
    self.cond = action.cond
    self.isramp = action.isramp
    if action.isramp:
        self.ramp = <object>action.ramp
    self.sync = action.sync
    self.reloc_id = action.reloc_id
    self.aid = action.aid
    self.tid = action.tid
    self.is_end = action.is_end
    self.seq_time = action.seq_time
    self.float_value = action.float_value
    self.bool_value = action.bool_value
    return self

cdef class ToneChannel:
    cdef public int chn
    cdef public list actions

cdef ToneChannel new_tone_channel(rfsoc_backend.ToneChannel *tone_chn):
    self = <ToneChannel>ToneChannel.__new__(ToneChannel)
    self.chn = tone_chn.chn
    self.actions = [[new_rfsoc_action(&tone_chn.actions[param][i])
                     for i in range(tone_chn.actions[param].size())]
                    for param in range(4)]
    return self

cdef class ChannelInfo:
    cdef public list channels
    cdef public dict chn_map

cdef param_names = {
    rfsoc_backend.ToneFreq: 'freq',
    rfsoc_backend.ToneAmp: 'amp',
    rfsoc_backend.TonePhase: 'phase',
    rfsoc_backend.ToneFF: 'ff'
}

cdef ChannelInfo new_channel_info(rfsoc_backend.ChannelInfo *info):
    self = <ChannelInfo>ChannelInfo.__new__(ChannelInfo)
    self.channels = [new_tone_channel(&info.channels[i])
                     for i in range(info.channels.size())]
    self.chn_map = {seq_chn: (chn_idx, param_names[param])
                        for seq_chn, (chn_idx, param) in info.chn_map}
    return self

def get_channel_info(rfsoc_backend.RFSOCBackend rb):
    return new_channel_info(&rb.channels)

cdef class Relocation:
    cdef public int cond_idx
    cdef public int time_idx
    cdef public int val_idx

cdef Relocation new_relocation(rfsoc_backend.Relocation c_reloc):
    py_reloc = <Relocation>Relocation.__new__(Relocation)
    py_reloc.cond_idx = c_reloc.cond_idx
    py_reloc.time_idx = c_reloc.time_idx
    py_reloc.val_idx = c_reloc.val_idx
    return py_reloc

class CompiledInfo:
    pass

def get_compiled_info(rfsoc_backend.RFSOCBackend rb):
    self = CompiledInfo()
    self.bool_values = [<object>p.first for p in rb.bool_values]
    self.float_values = [<object>p.first for p in rb.float_values]
    self.relocations = [new_relocation(action) for action in rb.relocations]
    return self

cdef class ErrorGenerator(rfsoc_backend.RFSOCOutputGenerator):
    cdef int add_tone_data(self, int channel, int tone, int64_t duration_cycles,
                           rfsoc_backend.cubic_spline_t frequency_hz,
                           rfsoc_backend.cubic_spline_t amplitude,
                           rfsoc_backend.cubic_spline_t phase_rad,
                           rfsoc_backend.output_flags_t flags) except -1:
        raise RuntimeError("AAABBBCCCC Error Generator DDDEEEFFF")
