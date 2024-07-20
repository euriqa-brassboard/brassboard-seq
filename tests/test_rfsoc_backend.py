#

import dummy_pulse_compiler
dummy_pulse_compiler.inject()

from brassboard_seq.config import Config
from brassboard_seq import backend, rfsoc_backend, seq
import brassboard_seq_rfsoc_backend_utils as rfsoc_utils

import pytest

global_conf = Config()
global_conf.add_supported_prefix('artiq')
global_conf.add_supported_prefix('rfsoc')

def new_seq_compiler(*args):
    s = seq.Seq(global_conf, *args)
    comp = backend.SeqCompiler(s)
    return s, comp

global_gen = rfsoc_backend.PulseCompilerGenerator()

def add_rfsoc_backend(comp):
    rb = rfsoc_backend.RFSOCBackend(global_gen)
    comp.add_backend('rfsoc', rb)
    comp.add_backend('artiq', backend.Backend()) # Dummy backend
    return rb

def with_params(*params):
    def deco(f):
        def wrapper():
            for param in params:
                f(*param)
        wrapper.__name__ = f.__name__
        return wrapper
    return deco

def test_generator():
    rfsoc_backend.PulseCompilerGenerator()
    rfsoc_backend.PulseCompilerGenerator()

def get_channel_info(rb):
    channels = rfsoc_utils.get_channel_info(rb)
    chn_ids = set(tone_chn.chn for tone_chn in channels.channels)
    assert len(chn_ids) == len(channels.channels)
    chn_params = set()
    for seq_chn, (chn_idx, param) in channels.chn_map.items():
        assert param in ('amp', 'freq', 'phase', 'ff')
        assert 0 <= chn_idx
        assert chn_idx < len(channels.channels)
        assert (chn_idx, param) not in chn_params
        chn_params.add((chn_idx, param))
    return channels

with_rfsoc_params = with_params((0,), (5,), (500,))

@with_rfsoc_params
def test_channels(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    assert s.get_channel_id('artiq/ttl0') == 0
    assert s.get_channel_id('rfsoc/dds1/0/amp') == 1
    assert s.get_channel_id('rfsoc/dds1/1/freq') == 2
    assert s.get_channel_id('artiq/ttl2') == 3
    assert s.get_channel_id('rfsoc/dds2/0/phase') == 4
    assert s.get_channel_id('rfsoc/dds0/0/ff') == 5
    assert s.get_channel_id('rfsoc/dds1/1/amp') == 6
    assert s.get_channel_id('rfsoc/dds1/1/ff') == 7
    assert s.get_channel_id('rfsoc/dds0/0/freq') == 8
    comp.finalize()
    channels = get_channel_info(rb)
    chn_ids = [tone_chn.chn for tone_chn in channels.channels]
    assert chn_ids == [2, 3, 4, 0]
    assert channels.chn_map == {1: (0, 'amp'), 2: (1, 'freq'),
                                4: (2, 'phase'), 5: (3, 'ff'),
                                6: (1, 'amp'), 7: (1, 'ff'),
                                8: (3, 'freq')}

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    assert s.get_channel_id('rfsoc/dds0/0') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds0/0'):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    assert s.get_channel_id('rfsoc/dds0/0/freq/a') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds0/0/freq/a'):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    assert s.get_channel_id('rfsoc/ch0/0/freq') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/ch0/0/freq'):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    assert s.get_channel_id('rfsoc/dds50/0/freq') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds50/0/freq'):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    assert s.get_channel_id('rfsoc/dds10/2/freq') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds10/2/freq'):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    assert s.get_channel_id('rfsoc/dds10/0/param') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds10/0/param'):
        comp.finalize()
