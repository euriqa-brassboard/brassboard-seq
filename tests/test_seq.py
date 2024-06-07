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

    with pytest.raises(ValueError, match="Time delay cannot be negative"):
        s.wait(-0.001)

    s.wait(0)
    t1 = s.end_time
    assert test_utils.event_time_id(t1) == 1
    assert str(t1) == 'T[0] + 0 ps'

    s.wait(0.001, cond=v2)
    t2 = s.end_time
    assert test_utils.event_time_id(t2) == 2
    assert str(t2) == f'T[1] + (1 ms; if bool({v2}))'

    c1.wait(0.002, cond=True)
    t3 = s.end_time
    assert test_utils.event_time_id(t3) == 3
    assert str(t3) == 'T[2] + 2 ms'

    c2.wait(0.0002)
    t4 = s.end_time
    assert test_utils.event_time_id(t4) == 4
    assert str(t4) == f'T[3] + (200 us; if {bv1})'

    c3.wait(1)
    t5 = s.end_time
    assert test_utils.event_time_id(t5) == 5
    assert str(t5) == 'T[4] + (1 s; if False)'
