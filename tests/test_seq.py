#

from brassboard_seq.config import Config
from brassboard_seq import seq
import brassboard_seq_test_utils as test_utils
import pytest

def test_seq():
    with pytest.raises(TypeError):
        seq.SeqInfo()
    with pytest.raises(TypeError):
        seq.TimeSeq()
    with pytest.raises(TypeError):
        seq.TimeStep()
    with pytest.raises(TypeError):
        seq.SubSeq()

    conf = Config()
    conf.add_supported_prefix('artiq')
    conf.add_channel_alias('test_ttl', 'artiq/ttl1')

    s = seq.Seq(conf)
    assert s.start_time is None
    assert str(s.end_time) == '0 ps'
    assert s.get_channel_id('artiq/ttl1') == 0
    assert s.get_channel_id('test_ttl') == 0
    assert s.get_channel_id('artiq/ttl3') == 1
    assert s.get_channel_id('artiq/ttl1') == 0
    assert s.get_channel_id('test_ttl') == 0
    assert s.get_channel_id('artiq/ttl3') == 1
    assert s.get_channel_id('artiq/ttl2') == 2
    assert test_utils.seq_get_channel_paths(s) == [('artiq', 'ttl1'), ('artiq', 'ttl3'),
                                                   ('artiq', 'ttl2')]
