#

import dummy_artiq
dummy_artiq.inject()

from brassboard_seq.config import Config
from brassboard_seq import artiq_backend, backend, seq
import brassboard_seq_test_utils as test_utils
import brassboard_seq_artiq_backend_utils as artiq_utils
import pytest

global_conf = Config()
global_conf.add_supported_prefix('artiq')
global_conf.add_supported_prefix('rfsoc')

def new_seq_compiler(*args):
    s = seq.Seq(global_conf, *args)
    comp = backend.SeqCompiler(s)
    return s, comp

def add_artiq_backend(comp, sys):
    ab = artiq_backend.ArtiqBackend(sys)
    comp.add_backend('artiq', ab)
    comp.add_backend('rfsoc', backend.Backend()) # Dummy backend
    return ab

def with_params(*params):
    def deco(f):
        def wrapper():
            for param in params:
                f(*param)
        wrapper.__name__ = f.__name__
        return wrapper
    return deco

def check_bt(exc, max_bt, *names):
    fnames = [tb.name for tb in exc.traceback]
    for name in names:
        if max_bt == 0:
            assert name not in fnames
        else:
            assert name in fnames

with_artiq_params = with_params((0,), (5,), (500,))

def test_constructors():
    with pytest.raises(TypeError):
        artiq_backend.ArtiqBackend()

def get_channel_info(ab):
    channels = artiq_utils.get_channel_info(ab)
    ttl_tgts = set(ttl.target for ttl in channels.ttlchns)
    assert len(ttl_tgts) == len(channels.ttlchns)
    assert len(ttl_tgts) == len(channels.ttl_chn_map)
    bus_chns = set(bus.channel for bus in channels.urukul_busses)
    assert len(bus_chns) == len(channels.urukul_busses)
    bus_ids = set(dds.bus_id for dds in channels.ddschns)
    assert bus_ids == set(range(len(bus_chns)))
    all_chip_selects = {i: set() for i in range(len(bus_chns))}
    for dds in channels.ddschns:
        chip_selects = all_chip_selects[dds.bus_id]
        assert dds.chip_select not in chip_selects
        chip_selects.add(dds.chip_select)
    assert sum(len(cs) for cs in all_chip_selects.values()) == len(channels.ddschns)
    return channels

@with_artiq_params
def test_channels(max_bt):
    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.DummyDaxSystem())
    assert s.get_channel_id('artiq/ttl0') == 0
    assert s.get_channel_id('artiq/ttl1') == 1
    assert s.get_channel_id('artiq/ttl10_counter') == 2
    assert s.get_channel_id('rfsoc/ch1/0/freq') == 3
    assert s.get_channel_id('artiq/urukul3_ch0/freq') == 4
    assert s.get_channel_id('artiq/urukul3_ch0/amp') == 5
    assert s.get_channel_id('artiq/urukul3_ch1/phase') == 6
    assert s.get_channel_id('artiq/urukul3_ch2/sw') == 7
    assert s.get_channel_id('artiq/urukul0_ch0/sw') == 8
    comp.finalize()
    channels = get_channel_info(ab)
    assert channels.ttl_chn_map == {0: 0, 1: 1, 2: 2, 7: 3, 8: 4}
    assert not channels.ttlchns[0].iscounter
    assert not channels.ttlchns[1].iscounter
    assert channels.ttlchns[2].iscounter
    assert channels.dds_param_chn_map == {4: (0, 0), 5: (0, 1), 6: (1, 2)}

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.HasEnvironment())
    assert s.get_channel_id('artiq/ttl0') == 0
    assert s.get_channel_id('artiq/urukul3_ch0/freq') == 1
    assert s.get_channel_id('artiq/ttl1') == 2
    assert s.get_channel_id('artiq/urukul3_ch0/amp') == 3
    assert s.get_channel_id('artiq/ttl10_counter') == 4
    assert s.get_channel_id('artiq/urukul3_ch1/phase') == 5
    assert s.get_channel_id('rfsoc/ch1/0/freq') == 6
    assert s.get_channel_id('artiq/urukul3_ch2/sw') == 7
    assert s.get_channel_id('artiq/urukul0_ch0/sw') == 8
    comp.finalize()
    channels = get_channel_info(ab)
    assert channels.ttl_chn_map == {0: 0, 2: 1, 4: 2, 7: 3, 8: 4}
    assert not channels.ttlchns[0].iscounter
    assert not channels.ttlchns[1].iscounter
    assert channels.ttlchns[2].iscounter
    assert channels.dds_param_chn_map == {1: (0, 0), 3: (0, 1), 5: (1, 2)}

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.HasEnvironment())
    s.get_channel_id('artiq/urukul0_ch0/a/b')
    with pytest.raises(ValueError, match="Invalid channel name artiq/urukul0_ch0/a/b"):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.HasEnvironment())
    s.get_channel_id('artiq/urukul0_ch0/aaa')
    with pytest.raises(ValueError, match="Invalid channel name artiq/urukul0_ch0/aaa"):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.HasEnvironment())
    s.get_channel_id('artiq/ttl0/a')
    with pytest.raises(ValueError, match="Invalid channel name artiq/ttl0/a"):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.HasEnvironment())
    s.get_channel_id('artiq/ttl0_counter/a')
    with pytest.raises(ValueError, match="Invalid channel name artiq/ttl0_counter/a"):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.HasEnvironment())
    s.get_channel_id('artiq/something')
    with pytest.raises(Exception):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.HasEnvironment())
    s.get_channel_id('artiq')
    with pytest.raises(ValueError, match="Invalid channel name artiq"):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.HasEnvironment())
    s.get_channel_id('artiq/dummy_dev')
    with pytest.raises(ValueError, match="Unsupported device: "):
        comp.finalize()
