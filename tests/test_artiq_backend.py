#

import dummy_artiq
dummy_artiq.inject()

from brassboard_seq.config import Config
from brassboard_seq import artiq_backend, seq
import brassboard_seq_test_utils as test_utils
import brassboard_seq_artiq_backend_utils as artiq_utils
import pytest

global_conf = Config()
global_conf.add_supported_prefix('artiq')
global_conf.add_supported_prefix('rfsoc')

def new_seq():
    return seq.new_seq(global_conf)

def test_constructors():
    with pytest.raises(TypeError):
        artiq_backend.ChannelsInfo()
    with pytest.raises(TypeError):
        artiq_backend.ArtiqBackend()

def check_channels(channels):
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

def test_channels():
    s = new_seq()
    assert s.get_channel_id('artiq/ttl0') == 0
    assert s.get_channel_id('artiq/ttl1') == 1
    assert s.get_channel_id('artiq/ttl10_counter') == 2
    assert s.get_channel_id('rfsoc/ch1/0/freq') == 3
    assert s.get_channel_id('artiq/urukul3_ch0/freq') == 4
    assert s.get_channel_id('artiq/urukul3_ch0/amp') == 5
    assert s.get_channel_id('artiq/urukul3_ch1/phase') == 6
    assert s.get_channel_id('artiq/urukul3_ch2/sw') == 7
    assert s.get_channel_id('artiq/urukul0_ch0/sw') == 8
    test_utils.seq_finalize(s)
    ab = artiq_backend.new_artiq_backend(dummy_artiq.DummyDaxSystem(), s)
    artiq_utils.process_seq(ab)
    channels = artiq_utils.get_channel_info(ab)
    assert channels.ttl_chn_map == {0: 0, 1: 1, 2: 2, 7: 3, 8: 4}
    assert not channels.ttlchns[0].iscounter
    assert not channels.ttlchns[1].iscounter
    assert channels.ttlchns[2].iscounter
    assert channels.dds_param_chn_map == {4: (0, 0), 5: (0, 1), 6: (1, 2)}
    check_channels(channels)

    s = new_seq()
    assert s.get_channel_id('artiq/ttl0') == 0
    assert s.get_channel_id('artiq/urukul3_ch0/freq') == 1
    assert s.get_channel_id('artiq/ttl1') == 2
    assert s.get_channel_id('artiq/urukul3_ch0/amp') == 3
    assert s.get_channel_id('artiq/ttl10_counter') == 4
    assert s.get_channel_id('artiq/urukul3_ch1/phase') == 5
    assert s.get_channel_id('rfsoc/ch1/0/freq') == 6
    assert s.get_channel_id('artiq/urukul3_ch2/sw') == 7
    assert s.get_channel_id('artiq/urukul0_ch0/sw') == 8
    test_utils.seq_finalize(s)
    ab = artiq_backend.new_artiq_backend(dummy_artiq.HasEnvironment(), s)
    artiq_utils.process_seq(ab)
    channels = artiq_utils.get_channel_info(ab)
    assert channels.ttl_chn_map == {0: 0, 2: 1, 4: 2, 7: 3, 8: 4}
    assert not channels.ttlchns[0].iscounter
    assert not channels.ttlchns[1].iscounter
    assert channels.ttlchns[2].iscounter
    assert channels.dds_param_chn_map == {1: (0, 0), 3: (0, 1), 5: (1, 2)}
    check_channels(channels)

    s = new_seq()
    s.get_channel_id('artiq/urukul0_ch0/a/b')
    test_utils.seq_finalize(s)
    ab = artiq_backend.new_artiq_backend(dummy_artiq.HasEnvironment(), s)
    with pytest.raises(ValueError, match="Invalid channel name artiq/urukul0_ch0/a/b"):
        artiq_utils.process_seq(ab)

    s = new_seq()
    s.get_channel_id('artiq/urukul0_ch0/aaa')
    test_utils.seq_finalize(s)
    ab = artiq_backend.new_artiq_backend(dummy_artiq.HasEnvironment(), s)
    with pytest.raises(ValueError, match="Invalid channel name artiq/urukul0_ch0/aaa"):
        artiq_utils.process_seq(ab)

    s = new_seq()
    s.get_channel_id('artiq/ttl0/a')
    test_utils.seq_finalize(s)
    ab = artiq_backend.new_artiq_backend(dummy_artiq.HasEnvironment(), s)
    with pytest.raises(ValueError, match="Invalid channel name artiq/ttl0/a"):
        artiq_utils.process_seq(ab)

    s = new_seq()
    s.get_channel_id('artiq/ttl0_counter/a')
    test_utils.seq_finalize(s)
    ab = artiq_backend.new_artiq_backend(dummy_artiq.HasEnvironment(), s)
    with pytest.raises(ValueError, match="Invalid channel name artiq/ttl0_counter/a"):
        artiq_utils.process_seq(ab)

    s = new_seq()
    s.get_channel_id('artiq/something')
    test_utils.seq_finalize(s)
    ab = artiq_backend.new_artiq_backend(dummy_artiq.HasEnvironment(), s)
    with pytest.raises(Exception):
        artiq_utils.process_seq(ab)

    s = new_seq()
    s.get_channel_id('artiq')
    test_utils.seq_finalize(s)
    ab = artiq_backend.new_artiq_backend(dummy_artiq.HasEnvironment(), s)
    with pytest.raises(ValueError, match="Invalid channel name artiq"):
        artiq_utils.process_seq(ab)

    s = new_seq()
    s.get_channel_id('artiq/dummy_dev')
    test_utils.seq_finalize(s)
    ab = artiq_backend.new_artiq_backend(dummy_artiq.HasEnvironment(), s)
    with pytest.raises(ValueError, match="Unsupported device: "):
        artiq_utils.process_seq(ab)
