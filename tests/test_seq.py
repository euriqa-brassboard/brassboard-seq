#

from brassboard_seq.config import Config
from brassboard_seq import seq, rtval
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
    with pytest.raises(TypeError):
        seq.ConditionalWrapper()

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

    c1 = s.conditional(True)
    assert test_utils.seq_get_cond(c1) is True

    v1 = rtval.new_extern(lambda: True)
    bv1 = rtval.convert_bool(v1)

    c2 = c1.conditional(bv1)
    assert test_utils.seq_get_cond(c2) is bv1

    c3 = c2.conditional(False)
    assert test_utils.seq_get_cond(c3) is False

    c4 = c3.conditional(v1)
    assert test_utils.seq_get_cond(c4) is False

    c5 = c2.conditional(True)
    assert test_utils.seq_get_cond(c5) is bv1

    v2 = rtval.new_extern(lambda: False)
    c6 = c2.conditional(v2)
    cc6 = test_utils.seq_get_cond(c6)
    assert str(cc6) == f'{bv1} & bool({v2})'
