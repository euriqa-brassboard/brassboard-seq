#

from brassboard_seq.config import Config
from brassboard_seq import seq, rtval, scan
import py_test_utils as test_utils
import pytest

def test_seq():
    conf = Config()
    conf.add_supported_prefix('artiq')
    conf.add_channel_alias('test_ttl', 'artiq/ttl1')

    with pytest.raises(ValueError, match="max_frame cannot be negative"):
        seq.Seq(conf, -2)

    s = seq.Seq(conf)
    assert str(s) == """Seq - T[0]
 T[0]: 0 ps
"""
    assert str(s) == repr(s)
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

    with pytest.raises(AssertionError, match="Assertion failed"):
        s.rt_assert(0)
    with pytest.raises(AssertionError, match="Some message"):
        s.rt_assert(0, "Some message")
    s.rt_assert(1)

    with pytest.raises(TypeError, match="SubSeq.conditional\\(\\) takes exactly one argument \\(0 given\\)"):
        s.conditional()

    with pytest.raises(TypeError, match="SubSeq.conditional\\(\\) takes exactly one argument \\(2 given\\)"):
        s.conditional(True, 1)

    c1 = s.conditional(True)
    assert test_utils.seq_get_cond(c1) is True
    assert str(s) == """Seq - T[0]
 T[0]: 0 ps
"""
    assert str(c1) == """ConditionalWrapper(True) for
  Seq - T[0]
   T[0]: 0 ps
"""
    assert str(c1) == repr(c1)

    v1 = test_utils.new_extern(lambda: True)
    bv1 = rtval.convert_bool(v1)

    c2 = c1.conditional(bv1)
    assert test_utils.seq_get_cond(c2) is bv1
    assert str(c2) == f"""ConditionalWrapper({bv1}) for
  Seq - T[0]
   T[0]: 0 ps
"""

    c3 = c2.conditional(False)
    assert test_utils.seq_get_cond(c3) is False
    assert str(c3) == """ConditionalWrapper(False) for
  Seq - T[0]
   T[0]: 0 ps
"""

    c4 = c3.conditional(v1)
    assert test_utils.seq_get_cond(c4) is False
    assert str(c4) == """ConditionalWrapper(False) for
  Seq - T[0]
   T[0]: 0 ps
"""

    c5 = c2.conditional(True)
    assert test_utils.seq_get_cond(c5) is bv1
    assert str(c5) == f"""ConditionalWrapper({bv1}) for
  Seq - T[0]
   T[0]: 0 ps
"""

    v2 = test_utils.new_extern(lambda: False)
    c6 = c2.conditional(v2)
    cc6 = test_utils.seq_get_cond(c6)
    assert str(cc6) == f'{bv1} & bool({v2})'
    assert str(c6) == f"""ConditionalWrapper({bv1} & bool({v2})) for
  Seq - T[0]
   T[0]: 0 ps
"""

    with pytest.raises(ValueError, match="Time delay cannot be negative"):
        s.wait(-0.001)
    with pytest.raises(TypeError, match="unexpected"):
        s.wait(0.001, unknown=False)

    s.wait(0)
    t1 = s.end_time
    assert test_utils.event_time_id(t1) == 1
    assert str(t1) == 'T[0] + 0 ps'
    assert str(s) == f"""Seq - T[1]
 T[0]: 0 ps
 T[1]: {t1}
"""

    s.wait(0.001, cond=v2)
    t2 = s.end_time
    assert test_utils.event_time_id(t2) == 2
    assert str(t2) == f'T[1] + (1 ms; if bool({v2}))'
    assert str(s) == f"""Seq - T[2]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
"""

    c1.wait(0.002, cond=True)
    t3 = s.end_time
    assert test_utils.event_time_id(t3) == 3
    assert str(t3) == 'T[2] + 2 ms'
    assert str(s) == f"""Seq - T[3]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
"""

    c2.wait(0.0002)
    t4 = s.end_time
    assert test_utils.event_time_id(t4) == 4
    assert str(t4) == f'T[3] + (200 us; if {bv1})'
    assert str(s) == f"""Seq - T[4]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
"""

    c3.wait(1)
    t5 = s.end_time
    assert test_utils.event_time_id(t5) == 5
    assert str(t5) == 'T[4] + (1 s; if False)'
    assert str(s) == f"""Seq - T[5]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
"""

    st1 = s.add_step(0.2)
    assert st1.start_time is t5
    assert st1.end_time is s.end_time
    t6 = s.end_time
    assert test_utils.event_time_id(t6) == 6
    assert str(t6) == 'T[5] + 200 ms'
    assert str(st1) == "TimeStep(0.2)@T[5]\n"
    assert str(st1) == repr(st1)
    assert str(s) == f"""Seq - T[6]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
  TimeStep(0.2)@T[5]
"""
    st1.set('artiq/ttl1', True, cond=True)
    assert str(st1) == """TimeStep(0.2)@T[5]
  artiq/ttl1: Set(True)
"""
    assert str(s) == f"""Seq - T[6]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
  TimeStep(0.2)@T[5]
    artiq/ttl1: Set(True)
"""
    with pytest.raises(ValueError, match="Multiple actions added for "
                       "the same channel at the same time on artiq/ttl1"):
        st1.set('artiq/ttl1', True, cond=False)

    st2 = s.add_background(10)
    assert st2.start_time is t6
    assert s.end_time is t6
    t7 = st2.end_time
    assert test_utils.event_time_id(t7) == 7
    assert str(t7) == 'T[6] + 10 s'
    st2.pulse('artiq/ttl2', True, cond=False)
    assert str(s) == f"""Seq - T[6]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
 T[7]: {t7}
  TimeStep(0.2)@T[5]
    artiq/ttl1: Set(True)
  TimeStep(10)@T[6]
    artiq/ttl2: Pulse(True, cond=False)
"""

    st3 = s.add_floating(0.1)
    t8 = st3.start_time
    t9 = st3.end_time
    assert s.end_time is t6
    assert test_utils.event_time_id(t8) == 8
    assert test_utils.event_time_id(t9) == 9
    assert str(t8) == '<floating>'
    assert str(t9) == 'T[8] + 100 ms'
    st3.pulse(1, v1, cond=bv1)
    assert str(s) == f"""Seq - T[6]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
 T[7]: {t7}
 T[8]: {t8}
 T[9]: {t9}
  TimeStep(0.2)@T[5]
    artiq/ttl1: Set(True)
  TimeStep(10)@T[6]
    artiq/ttl2: Pulse(True, cond=False)
  TimeStep(0.1)@T[8]
    artiq/ttl3: Pulse({v1}, cond={bv1})
"""

    st4 = s.add_at(t9, 0.12)
    assert st4.start_time is t9
    t10 = st4.end_time
    assert s.end_time is t6
    assert test_utils.event_time_id(t10) == 10
    assert str(t10) == 'T[9] + 120 ms'
    st4.set('artiq/ttl4', v2, exact_time=True)
    assert s.get_channel_id('artiq/ttl4') == 3
    assert str(s) == f"""Seq - T[6]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
 T[7]: {t7}
 T[8]: {t8}
 T[9]: {t9}
 T[10]: {t10}
  TimeStep(0.2)@T[5]
    artiq/ttl1: Set(True)
  TimeStep(10)@T[6]
    artiq/ttl2: Pulse(True, cond=False)
  TimeStep(0.1)@T[8]
    artiq/ttl3: Pulse({v1}, cond={bv1})
  TimeStep(0.12)@T[9]
    artiq/ttl4: Set({v2}, exact_time=True)
"""

    with pytest.raises(ValueError, match="Time delay cannot be negative"):
        st3.set_time(t7, -0.1)

    st3.set_time(t7)
    assert str(t8) == 'T[7] + 0 ps'
    assert str(s) == f"""Seq - T[6]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
 T[7]: {t7}
 T[8]: {t8}
 T[9]: {t9}
 T[10]: {t10}
  TimeStep(0.2)@T[5]
    artiq/ttl1: Set(True)
  TimeStep(10)@T[6]
    artiq/ttl2: Pulse(True, cond=False)
  TimeStep(0.1)@T[8]
    artiq/ttl3: Pulse({v1}, cond={bv1})
  TimeStep(0.12)@T[9]
    artiq/ttl4: Set({v2}, exact_time=True)
"""

    st5 = c2.add_step(0.2)
    assert st5.start_time is t6
    assert st5.end_time is s.end_time
    t11 = s.end_time
    assert test_utils.event_time_id(t11) == 11
    assert str(t11) == f'T[6] + (200 ms; if {bv1})'
    f1 = test_utils.StaticFunction()
    st5.pulse('artiq/analog', f1, random_keyword=123)
    assert str(s) == f"""Seq - T[11]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
 T[7]: {t7}
 T[8]: {t8}
 T[9]: {t9}
 T[10]: {t10}
 T[11]: {t11}
  TimeStep(0.2)@T[5]
    artiq/ttl1: Set(True)
  TimeStep(10)@T[6]
    artiq/ttl2: Pulse(True, cond=False)
  TimeStep(0.1)@T[8]
    artiq/ttl3: Pulse({v1}, cond={bv1})
  TimeStep(0.12)@T[9]
    artiq/ttl4: Set({v2}, exact_time=True)
  TimeStep(0.2)@T[6] if {bv1}
    artiq/analog: Pulse({f1}, cond={bv1}, random_keyword=123)
"""

    st6 = c2.add_background(10)
    assert st6.start_time is t11
    assert s.end_time is t11
    t12 = st6.end_time
    assert test_utils.event_time_id(t12) == 12
    assert str(t12) == f'T[11] + (10 s; if {bv1})'
    assert str(s) == f"""Seq - T[11]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
 T[7]: {t7}
 T[8]: {t8}
 T[9]: {t9}
 T[10]: {t10}
 T[11]: {t11}
 T[12]: {t12}
  TimeStep(0.2)@T[5]
    artiq/ttl1: Set(True)
  TimeStep(10)@T[6]
    artiq/ttl2: Pulse(True, cond=False)
  TimeStep(0.1)@T[8]
    artiq/ttl3: Pulse({v1}, cond={bv1})
  TimeStep(0.12)@T[9]
    artiq/ttl4: Set({v2}, exact_time=True)
  TimeStep(0.2)@T[6] if {bv1}
    artiq/analog: Pulse({f1}, cond={bv1}, random_keyword=123)
  TimeStep(10)@T[11] if {bv1}
"""

    st7 = c2.add_floating(0.1)
    t13 = st7.start_time
    t14 = st7.end_time
    assert s.end_time is t11
    assert test_utils.event_time_id(t13) == 13
    assert test_utils.event_time_id(t14) == 14
    assert str(t13) == '<floating>'
    assert str(t14) == f'T[13] + (100 ms; if {bv1})'
    f2 = test_utils.StaticFunction()
    st7.set('artiq/analog', f2)
    assert str(s) == f"""Seq - T[11]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
 T[7]: {t7}
 T[8]: {t8}
 T[9]: {t9}
 T[10]: {t10}
 T[11]: {t11}
 T[12]: {t12}
 T[13]: {t13}
 T[14]: {t14}
  TimeStep(0.2)@T[5]
    artiq/ttl1: Set(True)
  TimeStep(10)@T[6]
    artiq/ttl2: Pulse(True, cond=False)
  TimeStep(0.1)@T[8]
    artiq/ttl3: Pulse({v1}, cond={bv1})
  TimeStep(0.12)@T[9]
    artiq/ttl4: Set({v2}, exact_time=True)
  TimeStep(0.2)@T[6] if {bv1}
    artiq/analog: Pulse({f1}, cond={bv1}, random_keyword=123)
  TimeStep(10)@T[11] if {bv1}
  TimeStep(0.1)@T[13] if {bv1}
    artiq/analog: Set({f2}, cond={bv1})
"""

    st7.set_time(t12, 0.01)
    assert str(t13) == f'T[12] + (10 ms; if {bv1})'
    assert str(s) == f"""Seq - T[11]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
 T[7]: {t7}
 T[8]: {t8}
 T[9]: {t9}
 T[10]: {t10}
 T[11]: {t11}
 T[12]: {t12}
 T[13]: {t13}
 T[14]: {t14}
  TimeStep(0.2)@T[5]
    artiq/ttl1: Set(True)
  TimeStep(10)@T[6]
    artiq/ttl2: Pulse(True, cond=False)
  TimeStep(0.1)@T[8]
    artiq/ttl3: Pulse({v1}, cond={bv1})
  TimeStep(0.12)@T[9]
    artiq/ttl4: Set({v2}, exact_time=True)
  TimeStep(0.2)@T[6] if {bv1}
    artiq/analog: Pulse({f1}, cond={bv1}, random_keyword=123)
  TimeStep(10)@T[11] if {bv1}
  TimeStep(0.1)@T[13] if {bv1}
    artiq/analog: Set({f2}, cond={bv1})
"""

    st8 = c2.add_at(t14, 0.12)
    assert st8.start_time is t14
    t15 = st8.end_time
    assert s.end_time is t11
    assert test_utils.event_time_id(t15) == 15
    assert str(t15) == f'T[14] + (120 ms; if {bv1})'
    assert str(s) == f"""Seq - T[11]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
 T[7]: {t7}
 T[8]: {t8}
 T[9]: {t9}
 T[10]: {t10}
 T[11]: {t11}
 T[12]: {t12}
 T[13]: {t13}
 T[14]: {t14}
 T[15]: {t15}
  TimeStep(0.2)@T[5]
    artiq/ttl1: Set(True)
  TimeStep(10)@T[6]
    artiq/ttl2: Pulse(True, cond=False)
  TimeStep(0.1)@T[8]
    artiq/ttl3: Pulse({v1}, cond={bv1})
  TimeStep(0.12)@T[9]
    artiq/ttl4: Set({v2}, exact_time=True)
  TimeStep(0.2)@T[6] if {bv1}
    artiq/analog: Pulse({f1}, cond={bv1}, random_keyword=123)
  TimeStep(10)@T[11] if {bv1}
  TimeStep(0.1)@T[13] if {bv1}
    artiq/analog: Set({f2}, cond={bv1})
  TimeStep(0.12)@T[14] if {bv1}
"""

    st9 = s.add_step(step1, 0.1, cond1=True)
    assert st9.start_time is t11
    assert st9.end_time is s.end_time
    t16 = s.end_time
    assert test_utils.event_time_id(t16) == 16
    assert str(t16) == 'T[11] + 100 ms'
    assert str(s) == f"""Seq - T[16]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
 T[7]: {t7}
 T[8]: {t8}
 T[9]: {t9}
 T[10]: {t10}
 T[11]: {t11}
 T[12]: {t12}
 T[13]: {t13}
 T[14]: {t14}
 T[15]: {t15}
 T[16]: {t16}
 T[17]: T[11] + 200 ms
  TimeStep(0.2)@T[5]
    artiq/ttl1: Set(True)
  TimeStep(10)@T[6]
    artiq/ttl2: Pulse(True, cond=False)
  TimeStep(0.1)@T[8]
    artiq/ttl3: Pulse({v1}, cond={bv1})
  TimeStep(0.12)@T[9]
    artiq/ttl4: Set({v2}, exact_time=True)
  TimeStep(0.2)@T[6] if {bv1}
    artiq/analog: Pulse({f1}, cond={bv1}, random_keyword=123)
  TimeStep(10)@T[11] if {bv1}
  TimeStep(0.1)@T[13] if {bv1}
    artiq/analog: Set({f2}, cond={bv1})
  TimeStep(0.12)@T[14] if {bv1}
  SubSeq@T[11] - T[16]
    TimeStep(0.1)@T[11]
    TimeStep(0.2)@T[11]
"""

    st10 = s.add_background(step2, 0.2, cond2=bv1)
    assert st10.start_time is t16
    assert s.end_time is t16
    t19 = st10.end_time
    assert test_utils.event_time_id(t19) == 19
    assert str(t19) == f'T[18] + (200 ms; if {bv1})'
    assert str(s) == f"""Seq - T[16]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
 T[7]: {t7}
 T[8]: {t8}
 T[9]: {t9}
 T[10]: {t10}
 T[11]: {t11}
 T[12]: {t12}
 T[13]: {t13}
 T[14]: {t14}
 T[15]: {t15}
 T[16]: {t16}
 T[17]: T[11] + 200 ms
 T[18]: T[16] + 0.4 s
 T[19]: {t19}
  TimeStep(0.2)@T[5]
    artiq/ttl1: Set(True)
  TimeStep(10)@T[6]
    artiq/ttl2: Pulse(True, cond=False)
  TimeStep(0.1)@T[8]
    artiq/ttl3: Pulse({v1}, cond={bv1})
  TimeStep(0.12)@T[9]
    artiq/ttl4: Set({v2}, exact_time=True)
  TimeStep(0.2)@T[6] if {bv1}
    artiq/analog: Pulse({f1}, cond={bv1}, random_keyword=123)
  TimeStep(10)@T[11] if {bv1}
  TimeStep(0.1)@T[13] if {bv1}
    artiq/analog: Set({f2}, cond={bv1})
  TimeStep(0.12)@T[14] if {bv1}
  SubSeq@T[11] - T[16]
    TimeStep(0.1)@T[11]
    TimeStep(0.2)@T[11]
  SubSeq@T[16] - T[19]
    TimeStep(0.4)@T[16]
    TimeStep(0.2)@T[18] if {bv1}
"""

    st11 = s.add_floating(step1, v1, cond1=False)
    t20 = st11.start_time
    t21 = st11.end_time
    assert s.end_time is t16
    assert test_utils.event_time_id(t20) == 20
    assert test_utils.event_time_id(t21) == 21
    assert str(t20) == '<floating>'
    assert str(t21) == f'T[20] + int64({v1} * 1000000000000)'
    assert str(s) == f"""Seq - T[16]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
 T[7]: {t7}
 T[8]: {t8}
 T[9]: {t9}
 T[10]: {t10}
 T[11]: {t11}
 T[12]: {t12}
 T[13]: {t13}
 T[14]: {t14}
 T[15]: {t15}
 T[16]: {t16}
 T[17]: T[11] + 200 ms
 T[18]: T[16] + 0.4 s
 T[19]: {t19}
 T[20]: {t20}
 T[21]: {t21}
 T[22]: T[20] + (int64({v1} * 2 * 1000000000000); if False)
  TimeStep(0.2)@T[5]
    artiq/ttl1: Set(True)
  TimeStep(10)@T[6]
    artiq/ttl2: Pulse(True, cond=False)
  TimeStep(0.1)@T[8]
    artiq/ttl3: Pulse({v1}, cond={bv1})
  TimeStep(0.12)@T[9]
    artiq/ttl4: Set({v2}, exact_time=True)
  TimeStep(0.2)@T[6] if {bv1}
    artiq/analog: Pulse({f1}, cond={bv1}, random_keyword=123)
  TimeStep(10)@T[11] if {bv1}
  TimeStep(0.1)@T[13] if {bv1}
    artiq/analog: Set({f2}, cond={bv1})
  TimeStep(0.12)@T[14] if {bv1}
  SubSeq@T[11] - T[16]
    TimeStep(0.1)@T[11]
    TimeStep(0.2)@T[11]
  SubSeq@T[16] - T[19]
    TimeStep(0.4)@T[16]
    TimeStep(0.2)@T[18] if {bv1}
  SubSeq@T[20] - T[21]
    TimeStep({v1})@T[20]
    TimeStep({v1} * 2)@T[20] if False
"""

    st11.set_time(t19, v1)
    assert str(t20) == f'T[19] + int64({v1} * 1000000000000)'
    assert str(s) == f"""Seq - T[16]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
 T[7]: {t7}
 T[8]: {t8}
 T[9]: {t9}
 T[10]: {t10}
 T[11]: {t11}
 T[12]: {t12}
 T[13]: {t13}
 T[14]: {t14}
 T[15]: {t15}
 T[16]: {t16}
 T[17]: T[11] + 200 ms
 T[18]: T[16] + 0.4 s
 T[19]: {t19}
 T[20]: {t20}
 T[21]: {t21}
 T[22]: T[20] + (int64({v1} * 2 * 1000000000000); if False)
  TimeStep(0.2)@T[5]
    artiq/ttl1: Set(True)
  TimeStep(10)@T[6]
    artiq/ttl2: Pulse(True, cond=False)
  TimeStep(0.1)@T[8]
    artiq/ttl3: Pulse({v1}, cond={bv1})
  TimeStep(0.12)@T[9]
    artiq/ttl4: Set({v2}, exact_time=True)
  TimeStep(0.2)@T[6] if {bv1}
    artiq/analog: Pulse({f1}, cond={bv1}, random_keyword=123)
  TimeStep(10)@T[11] if {bv1}
  TimeStep(0.1)@T[13] if {bv1}
    artiq/analog: Set({f2}, cond={bv1})
  TimeStep(0.12)@T[14] if {bv1}
  SubSeq@T[11] - T[16]
    TimeStep(0.1)@T[11]
    TimeStep(0.2)@T[11]
  SubSeq@T[16] - T[19]
    TimeStep(0.4)@T[16]
    TimeStep(0.2)@T[18] if {bv1}
  SubSeq@T[20] - T[21]
    TimeStep({v1})@T[20]
    TimeStep({v1} * 2)@T[20] if False
"""

    st12 = s.add_at(t21, step2, v2, cond2=False)
    assert st12.start_time is t21
    t24 = st12.end_time
    assert s.end_time is t16
    assert test_utils.event_time_id(t24) == 24
    assert str(t24) == f'T[23] + (int64({v2} * 1000000000000); if False)'
    assert str(s) == f"""Seq - T[16]
 T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
 T[7]: {t7}
 T[8]: {t8}
 T[9]: {t9}
 T[10]: {t10}
 T[11]: {t11}
 T[12]: {t12}
 T[13]: {t13}
 T[14]: {t14}
 T[15]: {t15}
 T[16]: {t16}
 T[17]: T[11] + 200 ms
 T[18]: T[16] + 0.4 s
 T[19]: {t19}
 T[20]: {t20}
 T[21]: {t21}
 T[22]: T[20] + (int64({v1} * 2 * 1000000000000); if False)
 T[23]: T[21] + int64({v2} * 2 * 1000000000000)
 T[24]: {t24}
  TimeStep(0.2)@T[5]
    artiq/ttl1: Set(True)
  TimeStep(10)@T[6]
    artiq/ttl2: Pulse(True, cond=False)
  TimeStep(0.1)@T[8]
    artiq/ttl3: Pulse({v1}, cond={bv1})
  TimeStep(0.12)@T[9]
    artiq/ttl4: Set({v2}, exact_time=True)
  TimeStep(0.2)@T[6] if {bv1}
    artiq/analog: Pulse({f1}, cond={bv1}, random_keyword=123)
  TimeStep(10)@T[11] if {bv1}
  TimeStep(0.1)@T[13] if {bv1}
    artiq/analog: Set({f2}, cond={bv1})
  TimeStep(0.12)@T[14] if {bv1}
  SubSeq@T[11] - T[16]
    TimeStep(0.1)@T[11]
    TimeStep(0.2)@T[11]
  SubSeq@T[16] - T[19]
    TimeStep(0.4)@T[16]
    TimeStep(0.2)@T[18] if {bv1}
  SubSeq@T[20] - T[21]
    TimeStep({v1})@T[20]
    TimeStep({v1} * 2)@T[20] if False
  SubSeq@T[21] - T[24]
    TimeStep({v2} * 2)@T[21]
    TimeStep({v2})@T[23] if False
"""

    s_t1_24 = f""" T[0]: 0 ps
 T[1]: {t1}
 T[2]: {t2}
 T[3]: {t3}
 T[4]: {t4}
 T[5]: {t5}
 T[6]: {t6}
 T[7]: {t7}
 T[8]: {t8}
 T[9]: {t9}
 T[10]: {t10}
 T[11]: {t11}
 T[12]: {t12}
 T[13]: {t13}
 T[14]: {t14}
 T[15]: {t15}
 T[16]: {t16}
 T[17]: T[11] + 200 ms
 T[18]: T[16] + 0.4 s
 T[19]: {t19}
 T[20]: {t20}
 T[21]: {t21}
 T[22]: T[20] + (int64({v1} * 2 * 1000000000000); if False)
 T[23]: T[21] + int64({v2} * 2 * 1000000000000)
 T[24]: {t24}"""
    s_st1_12 = f"""  TimeStep(0.2)@T[5]
    artiq/ttl1: Set(True)
  TimeStep(10)@T[6]
    artiq/ttl2: Pulse(True, cond=False)
  TimeStep(0.1)@T[8]
    artiq/ttl3: Pulse({v1}, cond={bv1})
  TimeStep(0.12)@T[9]
    artiq/ttl4: Set({v2}, exact_time=True)
  TimeStep(0.2)@T[6] if {bv1}
    artiq/analog: Pulse({f1}, cond={bv1}, random_keyword=123)
  TimeStep(10)@T[11] if {bv1}
  TimeStep(0.1)@T[13] if {bv1}
    artiq/analog: Set({f2}, cond={bv1})
  TimeStep(0.12)@T[14] if {bv1}
  SubSeq@T[11] - T[16]
    TimeStep(0.1)@T[11]
    TimeStep(0.2)@T[11]
  SubSeq@T[16] - T[19]
    TimeStep(0.4)@T[16]
    TimeStep(0.2)@T[18] if {bv1}
  SubSeq@T[20] - T[21]
    TimeStep({v1})@T[20]
    TimeStep({v1} * 2)@T[20] if False
  SubSeq@T[21] - T[24]
    TimeStep({v2} * 2)@T[21]
    TimeStep({v2})@T[23] if False"""

    st13 = c2.add_step(step1, 0.1, cond1=True)
    assert st13.start_time is t16
    assert st13.end_time is s.end_time
    t25 = s.end_time
    assert test_utils.event_time_id(t25) == 25
    assert str(t25) == f'T[16] + (100 ms; if {bv1})'
    assert str(s) == f"""Seq - T[25]
{s_t1_24}
 T[25]: {t25}
 T[26]: T[16] + (200 ms; if {bv1})
{s_st1_12}
  SubSeq@T[16] - T[25] if {bv1}
    TimeStep(0.1)@T[16] if {bv1}
    TimeStep(0.2)@T[16] if {bv1}
"""

    st14 = c2.add_background(step2, 0.2, cond2=v2)
    assert st14.start_time is t25
    assert s.end_time is t25
    t28 = st14.end_time
    assert test_utils.event_time_id(t28) == 28
    assert str(t28) == f'T[27] + (200 ms; if {bv1} & bool({v2}))'
    assert str(s) == f"""Seq - T[25]
{s_t1_24}
 T[25]: {t25}
 T[26]: T[16] + (200 ms; if {bv1})
 T[27]: T[25] + (0.4 s; if {bv1})
 T[28]: {t28}
{s_st1_12}
  SubSeq@T[16] - T[25] if {bv1}
    TimeStep(0.1)@T[16] if {bv1}
    TimeStep(0.2)@T[16] if {bv1}
  SubSeq@T[25] - T[28] if {bv1}
    TimeStep(0.4)@T[25] if {bv1}
    TimeStep(0.2)@T[27] if {bv1} & bool({v2})
"""

    st15 = c2.add_floating(step1, v1, cond1=False)
    t29 = st15.start_time
    t30 = st15.end_time
    assert s.end_time is t25
    assert test_utils.event_time_id(t29) == 29
    assert test_utils.event_time_id(t30) == 30
    assert str(t29) == '<floating>'
    assert str(t30) == f'T[29] + (int64({v1} * 1000000000000); if {bv1})'
    assert str(s) == f"""Seq - T[25]
{s_t1_24}
 T[25]: {t25}
 T[26]: T[16] + (200 ms; if {bv1})
 T[27]: T[25] + (0.4 s; if {bv1})
 T[28]: {t28}
 T[29]: {t29}
 T[30]: {t30}
 T[31]: T[29] + (int64({v1} * 2 * 1000000000000); if False)
{s_st1_12}
  SubSeq@T[16] - T[25] if {bv1}
    TimeStep(0.1)@T[16] if {bv1}
    TimeStep(0.2)@T[16] if {bv1}
  SubSeq@T[25] - T[28] if {bv1}
    TimeStep(0.4)@T[25] if {bv1}
    TimeStep(0.2)@T[27] if {bv1} & bool({v2})
  SubSeq@T[29] - T[30] if {bv1}
    TimeStep({v1})@T[29] if {bv1}
    TimeStep({v1} * 2)@T[29] if False
"""

    st15.set_time(t28, v2)
    assert str(t29) == f'T[28] + (int64({v2} * 1000000000000); if {bv1})'
    assert str(s) == f"""Seq - T[25]
{s_t1_24}
 T[25]: {t25}
 T[26]: T[16] + (200 ms; if {bv1})
 T[27]: T[25] + (0.4 s; if {bv1})
 T[28]: {t28}
 T[29]: {t29}
 T[30]: {t30}
 T[31]: T[29] + (int64({v1} * 2 * 1000000000000); if False)
{s_st1_12}
  SubSeq@T[16] - T[25] if {bv1}
    TimeStep(0.1)@T[16] if {bv1}
    TimeStep(0.2)@T[16] if {bv1}
  SubSeq@T[25] - T[28] if {bv1}
    TimeStep(0.4)@T[25] if {bv1}
    TimeStep(0.2)@T[27] if {bv1} & bool({v2})
  SubSeq@T[29] - T[30] if {bv1}
    TimeStep({v1})@T[29] if {bv1}
    TimeStep({v1} * 2)@T[29] if False
"""

    st16 = c2.add_at(t30, step2, v2, cond2=False)
    assert st16.start_time is t30
    t33 = st16.end_time
    assert s.end_time is t25
    assert test_utils.event_time_id(t33) == 33
    assert str(t33) == f'T[32] + (int64({v2} * 1000000000000); if False)'
    assert str(s) == f"""Seq - T[25]
{s_t1_24}
 T[25]: {t25}
 T[26]: T[16] + (200 ms; if {bv1})
 T[27]: T[25] + (0.4 s; if {bv1})
 T[28]: {t28}
 T[29]: {t29}
 T[30]: {t30}
 T[31]: T[29] + (int64({v1} * 2 * 1000000000000); if False)
 T[32]: T[30] + (int64({v2} * 2 * 1000000000000); if {bv1})
 T[33]: {t33}
{s_st1_12}
  SubSeq@T[16] - T[25] if {bv1}
    TimeStep(0.1)@T[16] if {bv1}
    TimeStep(0.2)@T[16] if {bv1}
  SubSeq@T[25] - T[28] if {bv1}
    TimeStep(0.4)@T[25] if {bv1}
    TimeStep(0.2)@T[27] if {bv1} & bool({v2})
  SubSeq@T[29] - T[30] if {bv1}
    TimeStep({v1})@T[29] if {bv1}
    TimeStep({v1} * 2)@T[29] if False
  SubSeq@T[30] - T[33] if {bv1}
    TimeStep({v2} * 2)@T[30] if {bv1}
    TimeStep({v2})@T[32] if False
"""

    s.wait_for(st14, 0.01)
    t34 = s.current_time
    assert test_utils.event_time_id(t34) == 34
    assert str(t34) == 'T[25]; wait_for(T[28] + 10 ms)'
    assert str(s) == f"""Seq - T[34]
{s_t1_24}
 T[25]: {t25}
 T[26]: T[16] + (200 ms; if {bv1})
 T[27]: T[25] + (0.4 s; if {bv1})
 T[28]: {t28}
 T[29]: {t29}
 T[30]: {t30}
 T[31]: T[29] + (int64({v1} * 2 * 1000000000000); if False)
 T[32]: T[30] + (int64({v2} * 2 * 1000000000000); if {bv1})
 T[33]: {t33}
 T[34]: {t34}
{s_st1_12}
  SubSeq@T[16] - T[25] if {bv1}
    TimeStep(0.1)@T[16] if {bv1}
    TimeStep(0.2)@T[16] if {bv1}
  SubSeq@T[25] - T[28] if {bv1}
    TimeStep(0.4)@T[25] if {bv1}
    TimeStep(0.2)@T[27] if {bv1} & bool({v2})
  SubSeq@T[29] - T[30] if {bv1}
    TimeStep({v1})@T[29] if {bv1}
    TimeStep({v1} * 2)@T[29] if False
  SubSeq@T[30] - T[33] if {bv1}
    TimeStep({v2} * 2)@T[30] if {bv1}
    TimeStep({v2})@T[32] if False
"""

    c2.wait_for(t33)
    t35 = s.current_time
    assert test_utils.event_time_id(t35) == 35
    assert str(t35) == f'T[34]; wait_for(T[33] + 0 ps; if {bv1})'
    assert str(s) == f"""Seq - T[35]
{s_t1_24}
 T[25]: {t25}
 T[26]: T[16] + (200 ms; if {bv1})
 T[27]: T[25] + (0.4 s; if {bv1})
 T[28]: {t28}
 T[29]: {t29}
 T[30]: {t30}
 T[31]: T[29] + (int64({v1} * 2 * 1000000000000); if False)
 T[32]: T[30] + (int64({v2} * 2 * 1000000000000); if {bv1})
 T[33]: {t33}
 T[34]: {t34}
 T[35]: {t35}
{s_st1_12}
  SubSeq@T[16] - T[25] if {bv1}
    TimeStep(0.1)@T[16] if {bv1}
    TimeStep(0.2)@T[16] if {bv1}
  SubSeq@T[25] - T[28] if {bv1}
    TimeStep(0.4)@T[25] if {bv1}
    TimeStep(0.2)@T[27] if {bv1} & bool({v2})
  SubSeq@T[29] - T[30] if {bv1}
    TimeStep({v1})@T[29] if {bv1}
    TimeStep({v1} * 2)@T[29] if False
  SubSeq@T[30] - T[33] if {bv1}
    TimeStep({v2} * 2)@T[30] if {bv1}
    TimeStep({v2})@T[32] if False
"""

    with pytest.raises(TypeError):
        s.wait_for(1)
    with pytest.raises(TypeError):
        s.wait_for(1, 0.1)
    with pytest.raises(TypeError):
        c2.wait_for(v1)
    with pytest.raises(TypeError):
        c2.wait_for(v2, 0.1)

    s.set('artiq/ttl1', False, cond=True)
    s.set('artiq/ttl2', False, exact_time=True)
    with pytest.raises(ValueError, match="Multiple actions added for "
                       "the same channel at the same time on artiq/ttl2"):
        s.set('artiq/ttl2', True)
    assert str(s) == f"""Seq - T[36]
{s_t1_24}
 T[25]: {t25}
 T[26]: T[16] + (200 ms; if {bv1})
 T[27]: T[25] + (0.4 s; if {bv1})
 T[28]: {t28}
 T[29]: {t29}
 T[30]: {t30}
 T[31]: T[29] + (int64({v1} * 2 * 1000000000000); if False)
 T[32]: T[30] + (int64({v2} * 2 * 1000000000000); if {bv1})
 T[33]: {t33}
 T[34]: {t34}
 T[35]: {t35}
 T[36]: T[35] + 0 ps
{s_st1_12}
  SubSeq@T[16] - T[25] if {bv1}
    TimeStep(0.1)@T[16] if {bv1}
    TimeStep(0.2)@T[16] if {bv1}
  SubSeq@T[25] - T[28] if {bv1}
    TimeStep(0.4)@T[25] if {bv1}
    TimeStep(0.2)@T[27] if {bv1} & bool({v2})
  SubSeq@T[29] - T[30] if {bv1}
    TimeStep({v1})@T[29] if {bv1}
    TimeStep({v1} * 2)@T[29] if False
  SubSeq@T[30] - T[33] if {bv1}
    TimeStep({v2} * 2)@T[30] if {bv1}
    TimeStep({v2})@T[32] if False
  TimeStep(0)@T[35]
    artiq/ttl1: Set(False)
    artiq/ttl2: Set(False, exact_time=True)
"""
    s.wait(0, cond=False)
    s.set('artiq/ttl2', True, kwarg=11)
    with pytest.raises(ValueError, match="Multiple actions added for "
                       "the same channel at the same time on artiq/ttl2"):
        c2.set('artiq/ttl2', True)
    c2.set('artiq/ttl5', True)
    c2.set('artiq/ttl6', True, cond=False)
    assert str(s) == f"""Seq - T[38]
{s_t1_24}
 T[25]: {t25}
 T[26]: T[16] + (200 ms; if {bv1})
 T[27]: T[25] + (0.4 s; if {bv1})
 T[28]: {t28}
 T[29]: {t29}
 T[30]: {t30}
 T[31]: T[29] + (int64({v1} * 2 * 1000000000000); if False)
 T[32]: T[30] + (int64({v2} * 2 * 1000000000000); if {bv1})
 T[33]: {t33}
 T[34]: {t34}
 T[35]: {t35}
 T[36]: T[35] + 0 ps
 T[37]: T[36] + (0 ps; if False)
 T[38]: T[37] + 0 ps
{s_st1_12}
  SubSeq@T[16] - T[25] if {bv1}
    TimeStep(0.1)@T[16] if {bv1}
    TimeStep(0.2)@T[16] if {bv1}
  SubSeq@T[25] - T[28] if {bv1}
    TimeStep(0.4)@T[25] if {bv1}
    TimeStep(0.2)@T[27] if {bv1} & bool({v2})
  SubSeq@T[29] - T[30] if {bv1}
    TimeStep({v1})@T[29] if {bv1}
    TimeStep({v1} * 2)@T[29] if False
  SubSeq@T[30] - T[33] if {bv1}
    TimeStep({v2} * 2)@T[30] if {bv1}
    TimeStep({v2})@T[32] if False
  TimeStep(0)@T[35]
    artiq/ttl1: Set(False)
    artiq/ttl2: Set(False, exact_time=True)
  TimeStep(0)@T[37]
    artiq/ttl2: Set(True, kwarg=11)
    artiq/ttl5: Set(True, cond={bv1})
    artiq/ttl6: Set(True, cond=False)
"""

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

    with pytest.raises(ValueError, match="Channel id -1 out of bound"):
        s.set(-1, True)
    with pytest.raises(ValueError, match="Channel id 100 out of bound"):
        s.set(100, True)

def step1(s, len1, *, cond1=None):
    t1 = s.start_time
    assert s.end_time is t1
    tid1 = test_utils.event_time_id(t1)

    step = s.add_step(len1)
    cc0 = test_utils.seq_get_cond(s)
    if cc0 is True:
        assert str(step) == f"TimeStep({len1})@T[{tid1}]\n"
    else:
        assert str(step) == f"TimeStep({len1})@T[{tid1}] if {cc0}\n"
    assert str(step) == repr(step)

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

    cc1 = test_utils.seq_get_cond(c1)
    if cc1 is True:
        assert str(step2) == f"TimeStep({len1 * 2})@T[{tid1}]\n"
    else:
        assert str(step2) == f"TimeStep({len1 * 2})@T[{tid1}] if {cc1}\n"
    assert str(step2) == repr(step2)
    if cc0 is True:
        assert str(s) == f"""SubSeq@T[{tid1}] - T[{tid2}]
  {step}  {step2}"""
        assert str(c1) == f"""ConditionalWrapper({cc1}) for
  SubSeq@T[{tid1}] - T[{tid2}]
    {step}    {step2}"""
    else:
        assert str(s) == f"""SubSeq@T[{tid1}] - T[{tid2}] if {cc0}
  {step}  {step2}"""
        assert str(c1) == f"""ConditionalWrapper({cc1}) for
  SubSeq@T[{tid1}] - T[{tid2}] if {cc0}
    {step}    {step2}"""
    assert str(s) == repr(s)
    assert str(c1) == repr(c1)

def step2(s, len2, *, cond2=None):
    t1 = s.start_time
    assert s.end_time is t1
    tid1 = test_utils.event_time_id(t1)

    step = s.add_step(len2 * 2)
    cc0 = test_utils.seq_get_cond(s)
    if cc0 is True:
        assert str(step) == f"TimeStep({len2 * 2})@T[{tid1}]\n"
    else:
        assert str(step) == f"TimeStep({len2 * 2})@T[{tid1}] if {cc0}\n"
    assert str(step) == repr(step)

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

    cc1 = test_utils.seq_get_cond(c1)
    if cc1 is True:
        assert str(step2) == f"TimeStep({len2})@T[{tid2}]\n"
    else:
        assert str(step2) == f"TimeStep({len2})@T[{tid2}] if {cc1}\n"
    assert str(step2) == repr(step2)
    if cc0 is True:
        assert str(s) == f"""SubSeq@T[{tid1}] - T[{tid3}]
  {step}  {step2}"""
        assert str(c1) == f"""ConditionalWrapper({cc1}) for
  SubSeq@T[{tid1}] - T[{tid3}]
    {step}    {step2}"""
    else:
        assert str(s) == f"""SubSeq@T[{tid1}] - T[{tid3}] if {cc0}
  {step}  {step2}"""
        assert str(c1) == f"""ConditionalWrapper({cc1}) for
  SubSeq@T[{tid1}] - T[{tid3}] if {cc0}
    {step}    {step2}"""
    assert str(s) == repr(s)
    assert str(c1) == repr(c1)

def test_cond_call():
    conf = Config()
    conf.add_supported_prefix('artiq')

    s = seq.Seq(conf)
    @s.conditional(False)
    def step(s):
        s.set('artiq/ttl0', True)

    assert str(s) == f"""Seq - T[1]
 T[0]: 0 ps
 T[1]: T[0] + (0 ps; if False)
  SubSeq@T[0] - T[1] if False
    TimeStep(0)@T[0] if False
      artiq/ttl0: Set(True, cond=False)
"""

def test_seq_arg_error():
    conf = Config()
    conf.add_supported_prefix('artiq')

    s = seq.Seq(conf)
    with pytest.raises(TypeError,
                       match='add_step\\(\\) takes at least 1 positional argument \\(0 given\\)'):
        s.add_step()
    with pytest.raises(TypeError,
                       match='add_background\\(\\) takes at least 1 positional argument \\(0 given\\)'):
        s.add_background()
    with pytest.raises(TypeError,
                       match='add_floating\\(\\) takes at least 1 positional argument \\(0 given\\)'):
        s.add_floating()
    with pytest.raises(TypeError,
                       match='add_at\\(\\) takes at least 2 positional arguments \\(0 given\\)'):
        s.add_at()
    with pytest.raises(TypeError,
                       match='add_at\\(\\) takes at least 2 positional arguments \\(1 given\\)'):
        s.add_at(1)
    with pytest.raises(TypeError,
                       match="Argument 'tp' has incorrect type"):
        s.add_at(1, 2)
    with pytest.raises(TypeError,
                       match='set\\(\\) takes exactly 2 positional arguments \\(0 given\\)'):
        s.set()
    with pytest.raises(TypeError,
                       match='set\\(\\) takes exactly 2 positional arguments \\(1 given\\)'):
        s.set(1)
    with pytest.raises(TypeError,
                       match='set\\(\\) takes exactly 2 positional arguments \\(3 given\\)'):
        s.set(1, 2, 3)
    c = s.conditional(True)
    with pytest.raises(TypeError,
                       match='add_step\\(\\) takes at least 1 positional argument \\(0 given\\)'):
        c.add_step()
    with pytest.raises(TypeError,
                       match='add_background\\(\\) takes at least 1 positional argument \\(0 given\\)'):
        c.add_background()
    with pytest.raises(TypeError,
                       match='add_floating\\(\\) takes at least 1 positional argument \\(0 given\\)'):
        c.add_floating()
    with pytest.raises(TypeError,
                       match='add_at\\(\\) takes at least 2 positional arguments \\(0 given\\)'):
        c.add_at()
    with pytest.raises(TypeError,
                       match='add_at\\(\\) takes at least 2 positional arguments \\(1 given\\)'):
        c.add_at(1)
    with pytest.raises(TypeError,
                       match="Argument 'tp' has incorrect type"):
        c.add_at(1, 2)
    with pytest.raises(TypeError,
                       match='set\\(\\) takes exactly 2 positional arguments \\(0 given\\)'):
        c.set()
    with pytest.raises(TypeError,
                       match='set\\(\\) takes exactly 2 positional arguments \\(1 given\\)'):
        c.set(1)
    with pytest.raises(TypeError,
                       match='set\\(\\) takes exactly 2 positional arguments \\(3 given\\)'):
        c.set(1, 2, 3)

    st = s.add_step(1)
    with pytest.raises(TypeError,
                       match='set\\(\\) takes exactly 2 positional arguments \\(0 given\\)'):
        st.set()
    with pytest.raises(TypeError,
                       match='set\\(\\) takes exactly 2 positional arguments \\(1 given\\)'):
        st.set(1)
    with pytest.raises(TypeError,
                       match='set\\(\\) takes exactly 2 positional arguments \\(3 given\\)'):
        st.set(1, 2, 3)
    with pytest.raises(TypeError,
                       match='pulse\\(\\) takes exactly 2 positional arguments \\(0 given\\)'):
        st.pulse()
    with pytest.raises(TypeError,
                       match='pulse\\(\\) takes exactly 2 positional arguments \\(1 given\\)'):
        st.pulse(1)
    with pytest.raises(TypeError,
                       match='pulse\\(\\) takes exactly 2 positional arguments \\(3 given\\)'):
        st.pulse(1, 2, 3)

def test_seq_C():
    conf = Config()

    s = seq.Seq(conf)
    assert isinstance(s.C, scan.ParamPack)
    assert s.C.a(1) == 1
    assert s.C[:] == dict(a=1)

    c = s.conditional(False)
    assert c.C is s.C

def test_invalid_branch():
    conf = Config()
    conf.add_supported_prefix('artiq')

    s = seq.Seq(conf)
    s2 = seq.Seq(conf)
    with pytest.raises(ValueError,
                       match="Cannot branch to basic seq from a different sequence"):
        s.add_branch(s2)

    s = seq.Seq(conf)
    bs1 = s.new_basic_seq()
    s.add_branch(bs1)
    with pytest.raises(ValueError, match="Branch already added"):
        s.add_branch(bs1)

def test_branch():
    conf = Config()
    conf.add_supported_prefix('artiq')

    s = seq.Seq(conf)
    s.add_step(2) \
      .set('artiq/ttl1', True) \
      .set('artiq/analog1', 2.3)
    s.wait(1)
    assert s.may_terminate

    bs1 = s.new_basic_seq()
    s.add_branch(bs1)
    assert not s.may_terminate
    assert bs1.may_terminate

    s.may_terminate = True
    assert s.may_terminate
    s.may_terminate = False
    assert not s.may_terminate
    bs1.add_step(0.1) \
      .pulse('artiq/ttl2', False) \
      .set('artiq/analog3', 0.2)
    bs1.wait(0.01)
    bs1.wait(0.02)

    assert str(bs1) == f"""BasicSeq[1] - T[3]
 T[0]: 0 ps
 T[1]: T[0] + 100 ms
 T[2]: T[1] + 10 ms
 T[3]: T[2] + 20 ms
  TimeStep(0.1)@T[0]
    artiq/ttl2: Pulse(False)
    artiq/analog3: Set(0.2)
"""
    assert str(bs1) == repr(bs1)

    assert str(s) == f"""Seq - T[2]
 branches: [1]
 T[0]: 0 ps
 T[1]: T[0] + 2 s
 T[2]: T[1] + 1 s
  TimeStep(2)@T[0]
    artiq/ttl1: Set(True)
    artiq/analog1: Set(2.3)

 BasicSeq[1] - T[3]
  T[0]: 0 ps
  T[1]: T[0] + 100 ms
  T[2]: T[1] + 10 ms
  T[3]: T[2] + 20 ms
   TimeStep(0.1)@T[0]
     artiq/ttl2: Pulse(False)
     artiq/analog3: Set(0.2)
"""

    s.may_terminate = True

    assert str(s) == f"""Seq - T[2]
 branches: [1] may terminate
 T[0]: 0 ps
 T[1]: T[0] + 2 s
 T[2]: T[1] + 1 s
  TimeStep(2)@T[0]
    artiq/ttl1: Set(True)
    artiq/analog1: Set(2.3)

 BasicSeq[1] - T[3]
  T[0]: 0 ps
  T[1]: T[0] + 100 ms
  T[2]: T[1] + 10 ms
  T[3]: T[2] + 20 ms
   TimeStep(0.1)@T[0]
     artiq/ttl2: Pulse(False)
     artiq/analog3: Set(0.2)
"""

    bs2 = s.new_basic_seq()
    s.add_branch(bs2)

    assert str(s) == f"""Seq - T[2]
 branches: [1 2] may terminate
 T[0]: 0 ps
 T[1]: T[0] + 2 s
 T[2]: T[1] + 1 s
  TimeStep(2)@T[0]
    artiq/ttl1: Set(True)
    artiq/analog1: Set(2.3)

 BasicSeq[1] - T[3]
  T[0]: 0 ps
  T[1]: T[0] + 100 ms
  T[2]: T[1] + 10 ms
  T[3]: T[2] + 20 ms
   TimeStep(0.1)@T[0]
     artiq/ttl2: Pulse(False)
     artiq/analog3: Set(0.2)

 BasicSeq[2] - T[0]
  T[0]: 0 ps
"""
