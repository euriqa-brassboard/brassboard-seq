# cython: language_level=3

from brassboard_seq cimport rfsoc_backend

cdef class ToneChannel:
    cdef public int chn

cdef ToneChannel new_tone_channel(rfsoc_backend.ToneChannel *tone_chn):
    self = <ToneChannel>ToneChannel.__new__(ToneChannel)
    self.chn = tone_chn.chn
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
