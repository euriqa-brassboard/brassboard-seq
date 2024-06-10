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
    assert s.current_time is s.end_time
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

    st1 = s.add_step(0.2)
    assert st1.start_time is t5
    assert st1.end_time is s.end_time
    t6 = s.end_time
    assert test_utils.event_time_id(t6) == 6
    assert str(t6) == 'T[5] + 200 ms'

    st2 = s.add_background(10)
    assert st2.start_time is t6
    assert s.end_time is t6
    t7 = st2.end_time
    assert test_utils.event_time_id(t7) == 7
    assert str(t7) == 'T[6] + 10 s'

    st3 = s.add_floating(0.1)
    t8 = st3.start_time
    t9 = st3.end_time
    assert s.end_time is t6
    assert test_utils.event_time_id(t8) == 8
    assert test_utils.event_time_id(t9) == 9
    assert str(t8) == '<floating>'
    assert str(t9) == 'T[8] + 100 ms'

    st4 = s.add_at(t9, 0.12)
    assert st4.start_time is t9
    t10 = st4.end_time
    assert s.end_time is t6
    assert test_utils.event_time_id(t10) == 10
    assert str(t10) == 'T[9] + 120 ms'

    with pytest.raises(ValueError, match="Time delay cannot be negative"):
        st3.set_time(t7, -0.1)

    st3.set_time(t7)
    assert str(t8) == 'T[7] + 0 ps'

    st5 = c2.add_step(0.2)
    assert st5.start_time is t6
    assert st5.end_time is s.end_time
    t11 = s.end_time
    assert test_utils.event_time_id(t11) == 11
    assert str(t11) == f'T[6] + (200 ms; if {bv1})'

    st6 = c2.add_background(10)
    assert st6.start_time is t11
    assert s.end_time is t11
    t12 = st6.end_time
    assert test_utils.event_time_id(t12) == 12
    assert str(t12) == f'T[11] + (10 s; if {bv1})'

    st7 = c2.add_floating(0.1)
    t13 = st7.start_time
    t14 = st7.end_time
    assert s.end_time is t11
    assert test_utils.event_time_id(t13) == 13
    assert test_utils.event_time_id(t14) == 14
    assert str(t13) == '<floating>'
    assert str(t14) == f'T[13] + (100 ms; if {bv1})'

    st7.set_time(t12, 0.01)
    assert str(t13) == f'T[12] + (10 ms; if {bv1})'

    st8 = c2.add_at(t14, 0.12)
    assert st8.start_time is t14
    t15 = st8.end_time
    assert s.end_time is t11
    assert test_utils.event_time_id(t15) == 15
    assert str(t15) == f'T[14] + (120 ms; if {bv1})'

    st9 = s.add_step(step1, 0.1, cond1=True)
    assert st9.start_time is t11
    assert st9.end_time is s.end_time
    t16 = s.end_time
    assert test_utils.event_time_id(t16) == 16
    assert str(t16) == 'T[11] + 100 ms'

    st10 = s.add_background(step2, 0.2, cond2=bv1)
    assert st10.start_time is t16
    assert s.end_time is t16
    t19 = st10.end_time
    assert test_utils.event_time_id(t19) == 19
    assert str(t19) == f'T[18] + (200 ms; if {bv1})'

    st11 = s.add_floating(step1, v1, cond1=False)
    t20 = st11.start_time
    t21 = st11.end_time
    assert s.end_time is t16
    assert test_utils.event_time_id(t20) == 20
    assert test_utils.event_time_id(t21) == 21
    assert str(t20) == '<floating>'
    assert str(t21) == f'T[20] + int64({v1} * 1000000000000)'

    st11.set_time(t19, v1)
    assert str(t20) == f'T[19] + int64({v1} * 1000000000000)'

    st12 = s.add_at(t21, step2, v2, cond2=False)
    assert st12.start_time is t21
    t24 = st12.end_time
    assert s.end_time is t16
    assert test_utils.event_time_id(t24) == 24
    assert str(t24) == f'T[23] + (int64({v2} * 1000000000000); if False)'

    st13 = c2.add_step(step1, 0.1, cond1=True)
    assert st13.start_time is t16
    assert st13.end_time is s.end_time
    t25 = s.end_time
    assert test_utils.event_time_id(t25) == 25
    assert str(t25) == f'T[16] + (100 ms; if {bv1})'

    st14 = c2.add_background(step2, 0.2, cond2=v2)
    assert st14.start_time is t25
    assert s.end_time is t25
    t28 = st14.end_time
    assert test_utils.event_time_id(t28) == 28
    assert str(t28) == f'T[27] + (200 ms; if {bv1} & bool({v2}))'

    st15 = c2.add_floating(step1, v1, cond1=False)
    t29 = st15.start_time
    t30 = st15.end_time
    assert s.end_time is t25
    assert test_utils.event_time_id(t29) == 29
    assert test_utils.event_time_id(t30) == 30
    assert str(t29) == '<floating>'
    assert str(t30) == f'T[29] + (int64({v1} * 1000000000000); if {bv1})'

    st15.set_time(t28, v2)
    assert str(t29) == f'T[28] + (int64({v2} * 1000000000000); if {bv1})'

    st16 = c2.add_at(t30, step2, v2, cond2=False)
    assert st16.start_time is t30
    t33 = st16.end_time
    assert s.end_time is t25
    assert test_utils.event_time_id(t33) == 33
    assert str(t33) == f'T[32] + (int64({v2} * 1000000000000); if False)'

    s.wait_for(st14, 0.01)
    t34 = s.current_time
    assert test_utils.event_time_id(t34) == 34
    assert str(t34) == 'T[25]; wait_for(T[28] + 10 ms)'

    c2.wait_for(t33)
    t35 = s.current_time
    assert test_utils.event_time_id(t35) == 35
    assert str(t35) == f'T[34]; wait_for(T[33] + 0 ps; if {bv1})'

    with pytest.raises(TypeError):
        s.wait_for(1)
    with pytest.raises(TypeError):
        s.wait_for(1, 0.1)
    with pytest.raises(TypeError):
        c2.wait_for(v1)
    with pytest.raises(TypeError):
        c2.wait_for(v2, 0.1)

def test_seq_error1():
    conf = Config()
    s = seq.Seq(conf)
    c1 = s.conditional(True)

    with pytest.raises(TypeError):
        s.add_step(step1)
    with pytest.raises(TypeError):
        s.add_step(step1, 1, 2)
    with pytest.raises(TypeError):
        s.add_step(step1, 1, cond2=True)

    with pytest.raises(TypeError):
        s.add_background(step1)
    with pytest.raises(TypeError):
        s.add_background(step1, 1, 2)
    with pytest.raises(TypeError):
        s.add_background(step1, 1, cond2=True)

    with pytest.raises(TypeError):
        s.add_floating(step1)
    with pytest.raises(TypeError):
        s.add_floating(step1, 1, 2)
    with pytest.raises(TypeError):
        s.add_floating(step1, 1, cond2=True)

    with pytest.raises(TypeError):
        s.add_at(s.start_time, step1)
    with pytest.raises(TypeError):
        s.add_at(s.start_time, step1, 1, 2)
    with pytest.raises(TypeError):
        s.add_at(s.start_time, step1, 1, cond2=True)

    with pytest.raises(ValueError, match="Unexpected arguments"):
        s.add_step(1, 2)
    with pytest.raises(ValueError, match="Unexpected arguments"):
        s.add_step(1, a=2)

    with pytest.raises(ValueError, match="Unexpected arguments"):
        s.add_background(1, 2)
    with pytest.raises(ValueError, match="Unexpected arguments"):
        s.add_background(1, a=2)

    with pytest.raises(ValueError, match="Unexpected arguments"):
        s.add_floating(1, 2)
    with pytest.raises(ValueError, match="Unexpected arguments"):
        s.add_floating(1, a=2)

    with pytest.raises(ValueError, match="Unexpected arguments"):
        s.add_at(s.start_time, 1, 2)
    with pytest.raises(ValueError, match="Unexpected arguments"):
        s.add_at(s.start_time, 1, a=2)

    with pytest.raises(TypeError):
        c1.add_step(step1)
    with pytest.raises(TypeError):
        c1.add_step(step1, 1, 2)
    with pytest.raises(TypeError):
        c1.add_step(step1, 1, cond2=True)

    with pytest.raises(TypeError):
        c1.add_background(step1)
    with pytest.raises(TypeError):
        c1.add_background(step1, 1, 2)
    with pytest.raises(TypeError):
        c1.add_background(step1, 1, cond2=True)

    with pytest.raises(TypeError):
        c1.add_floating(step1)
    with pytest.raises(TypeError):
        c1.add_floating(step1, 1, 2)
    with pytest.raises(TypeError):
        c1.add_floating(step1, 1, cond2=True)

    with pytest.raises(TypeError):
        c1.add_at(s.start_time, step1)
    with pytest.raises(TypeError):
        c1.add_at(s.start_time, step1, 1, 2)
    with pytest.raises(TypeError):
        c1.add_at(s.start_time, step1, 1, cond2=True)

    with pytest.raises(ValueError, match="Unexpected arguments"):
        c1.add_step(1, 2)
    with pytest.raises(ValueError, match="Unexpected arguments"):
        c1.add_step(1, a=2)

    with pytest.raises(ValueError, match="Unexpected arguments"):
        c1.add_background(1, 2)
    with pytest.raises(ValueError, match="Unexpected arguments"):
        c1.add_background(1, a=2)

    with pytest.raises(ValueError, match="Unexpected arguments"):
        c1.add_floating(1, 2)
    with pytest.raises(ValueError, match="Unexpected arguments"):
        c1.add_floating(1, a=2)

    with pytest.raises(ValueError, match="Unexpected arguments"):
        c1.add_at(s.start_time, 1, 2)
    with pytest.raises(ValueError, match="Unexpected arguments"):
        c1.add_at(s.start_time, 1, a=2)

def step1(s, len1, *, cond1=None):
    t1 = s.start_time
    assert s.end_time is t1
    tid1 = test_utils.event_time_id(t1)

    step = s.add_step(len1)

    assert step.start_time is t1
    t2 = step.end_time
    tid2 = test_utils.event_time_id(t2)
    assert tid2 >= tid1 + 1
    assert s.end_time is t2

    c1 = s.conditional(cond1)
    step2 = c1.add_at(t1, len1 * 2)

    assert step2.start_time is t1
    assert s.end_time is t2
    t3 = step2.end_time
    tid3 = test_utils.event_time_id(t3)
    assert tid3 == tid2 + 1

def step2(s, len2, *, cond2=None):
    t1 = s.start_time
    assert s.end_time is t1
    tid1 = test_utils.event_time_id(t1)

    step = s.add_step(len2 * 2)

    assert step.start_time is t1
    t2 = step.end_time
    tid2 = test_utils.event_time_id(t2)
    assert tid2 >= tid1 + 1
    assert s.end_time is t2

    c1 = s.conditional(cond2)
    step2 = c1.add_step(len2)

    assert step2.start_time is t2
    t3 = step2.end_time
    tid3 = test_utils.event_time_id(t3)
    assert tid3 == tid2 + 1
    assert s.end_time is t3
