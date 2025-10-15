#

from brassboard_seq.config import Config
from brassboard_seq import seq, rtval, action, backend
import py_test_utils as test_utils
import pytest

class Env:
    def __init__(self):
        self.conf = Config()
        self.conf.add_supported_prefix('artiq')

    def new_comp(self, *args):
        s = seq.Seq(self.conf, *args)
        comp = backend.SeqCompiler(s)
        comp.add_backend('artiq', backend.Backend()) # Dummy backend
        return comp

test_env = Env()

def test_compiler():
    s = seq.Seq(test_env.conf)
    with pytest.raises(TypeError, match="SeqCompiler.__init__ got an unexpected keyword argument 'a'"):
        backend.SeqCompiler(s, a=True)

@test_utils.with_seq_params
def test_cond_order1(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq

    c1val = True
    c1 = test_utils.new_extern(lambda: c1val)
    c2val = True
    c2 = test_utils.new_extern(lambda: c2val)

    def step_aaa_yyy_1(s):
        s.wait(0.1)
        s.conditional(c1).add_step(1).set('artiq/ttl1', True)
    st1 = s.add_background(step_aaa_yyy_1)
    def step_bbb_yxz_2(s):
        s.conditional(c2).wait_for(st1)
        s.add_step(2).set('artiq/ttl1', False)
    st2 = s.add_step(step_bbb_yxz_2)

    assert str(s) == f"""Seq - T[4]
 T[0]: 0 ps
 T[1]: T[0] + 100 ms
 T[2]: T[1] + (1 s; if bool({c1}))
 T[3]: T[0]; wait_for(T[2] + 0 ps; if bool({c2}))
 T[4]: T[3] + 2 s
  SubSeq@T[0] - T[2]
    TimeStep(1)@T[1] if bool({c1})
      artiq/ttl1: Set(True, cond=bool({c1}))
  SubSeq@T[0] - T[4]
    TimeStep(2)@T[3]
      artiq/ttl1: Set(False)
"""

    comp.finalize()

    assert str(s) == f"""Seq - T[4]
 T[0]: 0 ps
 T[1]: 100 ms
 T[2]: T[1] + (1 s; if bool({c1}))
 T[3]: T[0]; wait_for(T[2] + 0 ps; if bool({c2}))
 T[4]: T[3] + 2 s
  SubSeq@T[0] - T[2]
    TimeStep(1)@T[1] if bool({c1})
      artiq/ttl1: Set(True, cond=bool({c1}))
  SubSeq@T[0] - T[4]
    TimeStep(2)@T[3]
      artiq/ttl1: Set(False)
"""

    assert test_utils.compiler_num_basic_seq(comp) == 1
    actions = test_utils.compiler_get_all_actions(comp, 0)
    assert len(actions) == 1
    assert len(actions[0]) == 2
    action1 = actions[0][0]
    action2 = actions[0][1]
    assert str(action1) == f'Set(True, cond=bool({c1}))'
    assert str(action2) == f'Set(False)'
    info1 = action1.get_compile_info()
    info2 = action2.get_compile_info()
    assert info1['tid'] == 1
    assert info1['end_tid'] == 2
    assert info1['length'] == 1
    assert str(info1['end_val']) == f'ifelse(bool({c1}), True, False)'

    assert info2['tid'] == 3
    assert info2['end_tid'] == 4
    assert info2['length'] == 2
    assert info2['end_val'] is False

    comp.runtime_finalize(1)
    assert action1.get_cond_val() is True
    assert action2.get_cond_val() is True
    total_time, times = test_utils.compiler_get_all_times(comp, 0)
    assert test_utils.compiler_get_channel_changed(comp) == [True]

    assert times == [0,
                     100_000_000_000,
                     1100_000_000_000,
                     1100_000_000_000,
                     3100_000_000_000]
    assert total_time == 3100_000_000_000

    c1val = False
    comp.runtime_finalize(2)
    assert action1.get_cond_val() is c1val
    assert action2.get_cond_val() is True
    total_time, times = test_utils.compiler_get_all_times(comp, 0)
    assert test_utils.compiler_get_channel_changed(comp) == [True]

    assert times == [0,
                     100_000_000_000,
                     100_000_000_000,
                     100_000_000_000,
                     2100_000_000_000]
    assert total_time == 2100_000_000_000

    c2val = False
    comp.runtime_finalize(3)
    assert action1.get_cond_val() is c1val
    assert action2.get_cond_val() is True
    total_time, times = test_utils.compiler_get_all_times(comp, 0)
    assert test_utils.compiler_get_channel_changed(comp) == [True]

    assert times == [0,
                     100_000_000_000,
                     100_000_000_000,
                     0,
                     2000_000_000_000]
    assert total_time == 2000_000_000_000

    c1val = True
    with pytest.raises(ValueError, match="Action time order violation") as exc:
        comp.runtime_finalize(4)
    test_utils.check_bt(exc, max_bt, 'step_bbb_yxz_2')

@test_utils.with_seq_params
def test_cond_order2(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq

    c1val = True
    c1 = test_utils.new_extern(lambda: c1val)
    c2val = True
    c2 = test_utils.new_extern(lambda: c2val)

    def f_0_21234_alsdf():
        return s.add_floating(2).set('artiq/ttl1', False)
    def g_123_asnbe():
        return f_0_21234_alsdf()
    st1 = g_123_asnbe()
    def step_bbb_yxz_1(s):
        s.wait(0.1)
        s.conditional(c1).add_step(1).set('artiq/ttl1', True)
    st2 = s.add_background(step_bbb_yxz_1)
    s.conditional(c2).wait_for(st2)
    st1.set_time(s.current_time)

    assert str(s) == f"""Seq - T[5]
 T[0]: 0 ps
 T[1]: T[5] + 0 ps
 T[2]: T[1] + 2 s
 T[3]: T[0] + 100 ms
 T[4]: T[3] + (1 s; if bool({c1}))
 T[5]: T[0]; wait_for(T[4] + 0 ps; if bool({c2}))
  TimeStep(2)@T[1]
    artiq/ttl1: Set(False)
  SubSeq@T[0] - T[4]
    TimeStep(1)@T[3] if bool({c1})
      artiq/ttl1: Set(True, cond=bool({c1}))
"""

    comp.finalize()

    assert str(s) == f"""Seq - T[3]
 T[0]: 0 ps
 T[1]: 100 ms
 T[2]: T[1] + (1 s; if bool({c1}))
 T[3]: T[0]; wait_for(T[2] + 0 ps; if bool({c2}))
 T[4]: T[3] + 0 ps
 T[5]: T[4] + 2 s
  TimeStep(2)@T[4]
    artiq/ttl1: Set(False)
  SubSeq@T[0] - T[2]
    TimeStep(1)@T[1] if bool({c1})
      artiq/ttl1: Set(True, cond=bool({c1}))
"""

    assert test_utils.compiler_num_basic_seq(comp) == 1
    actions = test_utils.compiler_get_all_actions(comp, 0)
    assert len(actions) == 1
    assert len(actions[0]) == 2
    action1 = actions[0][0]
    action2 = actions[0][1]
    assert str(action1) == f'Set(True, cond=bool({c1}))'
    assert str(action2) == f'Set(False)'
    info1 = action1.get_compile_info()
    info2 = action2.get_compile_info()
    assert info1['tid'] == 1
    assert info1['end_tid'] == 2
    assert info1['length'] == 1
    assert str(info1['end_val']) == f'ifelse(bool({c1}), True, False)'

    assert info2['tid'] == 4
    assert info2['end_tid'] == 5
    assert info2['length'] == 2
    assert info2['end_val'] is False

    comp.runtime_finalize(1)
    assert action1.get_cond_val() is True
    assert action2.get_cond_val() is True
    total_time, times = test_utils.compiler_get_all_times(comp, 0)
    assert test_utils.compiler_get_channel_changed(comp) == [True]

    assert times == [0,
                     100_000_000_000,
                     1100_000_000_000,
                     1100_000_000_000,
                     1100_000_000_000,
                     3100_000_000_000]
    assert total_time == 3100_000_000_000

    c1val = False
    comp.runtime_finalize(2)
    assert action1.get_cond_val() is c1val
    assert action2.get_cond_val() is True
    total_time, times = test_utils.compiler_get_all_times(comp, 0)
    assert test_utils.compiler_get_channel_changed(comp) == [True]

    assert times == [0,
                     100_000_000_000,
                     100_000_000_000,
                     100_000_000_000,
                     100_000_000_000,
                     2100_000_000_000]
    assert total_time == 2100_000_000_000

    c2val = False
    comp.runtime_finalize(3)
    assert action1.get_cond_val() is c1val
    assert action2.get_cond_val() is True
    total_time, times = test_utils.compiler_get_all_times(comp, 0)
    assert test_utils.compiler_get_channel_changed(comp) == [True]

    assert times == [0,
                     100_000_000_000,
                     100_000_000_000,
                     0,
                     0,
                     2000_000_000_000]
    assert total_time == 2000_000_000_000

    c1val = True
    with pytest.raises(ValueError, match="Action time order violation") as exc:
        comp.runtime_finalize(4)
    test_utils.check_bt(exc, max_bt, 'g_123_asnbe', 'f_0_21234_alsdf')

@test_utils.with_seq_params
def test_order_error1(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq

    c1val = True
    c1 = test_utils.new_extern(lambda: c1val)

    def f1_89374asjkl_asbd():
        st1 = s.conditional(c1).add_background(1).set('artiq/ttl1', True)
        st2 = s.conditional(rtval.inv(c1)).add_background(2).set('artiq/ttl1', True)
    def gj18j_9034rfj():
        f1_89374asjkl_asbd()
    gj18j_9034rfj()

    assert str(s) == f"""Seq - T[0]
 T[0]: 0 ps
 T[1]: T[0] + (1 s; if bool({c1}))
 T[2]: T[0] + (2 s; if inv({c1}))
  TimeStep(1)@T[0] if bool({c1})
    artiq/ttl1: Set(True, cond=bool({c1}))
  TimeStep(2)@T[0] if inv({c1})
    artiq/ttl1: Set(True, cond=inv({c1}))
"""

    with pytest.raises(ValueError,
                       match="Multiple actions added for the same channel "
                       "at the same time on artiq/ttl1") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'f1_89374asjkl_asbd', 'gj18j_9034rfj')

@test_utils.with_seq_params
def test_order_error2(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq

    c1val = True
    c1 = test_utils.new_extern(lambda: c1val)

    def step_13238324(s):
        s.wait(0.1)
        def f_1239():
            s.set('artiq/ttl1', True)
        f_1239()
    s.conditional(c1).add_background(step_13238324)
    def kjaksd9123():
        s.conditional(rtval.inv(c1)).add_background(step_13238324)
    kjaksd9123()

    assert str(s) == f"""Seq - T[0]
 T[0]: 0 ps
 T[1]: T[0] + (100 ms; if bool({c1}))
 T[2]: T[1] + (0 ps; if bool({c1}))
 T[3]: T[0] + (100 ms; if inv({c1}))
 T[4]: T[3] + (0 ps; if inv({c1}))
  SubSeq@T[0] - T[2] if bool({c1})
    TimeStep(0)@T[1] if bool({c1})
      artiq/ttl1: Set(True, cond=bool({c1}))
  SubSeq@T[0] - T[4] if inv({c1})
    TimeStep(0)@T[3] if inv({c1})
      artiq/ttl1: Set(True, cond=inv({c1}))
"""

    with pytest.raises(ValueError,
                       match="Actions on artiq/ttl1 is not statically ordered") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'f_1239', 'step_13238324', 'kjaksd9123')

@test_utils.with_seq_params
def test_order_error3(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.add_background(0.1).set('artiq/ttl1', True)
    s.wait(0.2)
    s.add_background(0.1).set('artiq/ttl1', False)

    assert str(s) == f"""Seq - T[2]
 T[0]: 0 ps
 T[1]: T[0] + 100 ms
 T[2]: T[0] + 200 ms
 T[3]: T[2] + 100 ms
  TimeStep(0.1)@T[0]
    artiq/ttl1: Set(True)
  TimeStep(0.1)@T[2]
    artiq/ttl1: Set(False)
"""

    comp.finalize()

    assert str(s) == f"""Seq - T[2]
 T[0]: 0 ps
 T[1]: 100 ms
 T[2]: 200 ms
 T[3]: 0.3 s
  TimeStep(0.1)@T[0]
    artiq/ttl1: Set(True)
  TimeStep(0.1)@T[2]
    artiq/ttl1: Set(False)
"""

    assert test_utils.compiler_num_basic_seq(comp) == 1
    actions = test_utils.compiler_get_all_actions(comp, 0)
    assert len(actions) == 1
    assert len(actions[0]) == 2
    action1 = actions[0][0]
    action2 = actions[0][1]
    assert str(action1) == f'Set(True)'
    assert str(action2) == f'Set(False)'
    info1 = action1.get_compile_info()
    info2 = action2.get_compile_info()
    assert info1['tid'] == 0
    assert info1['end_tid'] == 1
    assert info1['length'] == 0.1
    assert info1['end_val'] is True

    assert info2['tid'] == 2
    assert info2['end_tid'] == 3
    assert info2['length'] == 0.1
    assert info2['end_val'] is False

    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.add_background(0.1).set('artiq/ttl1', True)
    s.wait(0.2)
    s.add_background(0.1).pulse('artiq/ttl1', False)

    assert str(s) == f"""Seq - T[2]
 T[0]: 0 ps
 T[1]: T[0] + 100 ms
 T[2]: T[0] + 200 ms
 T[3]: T[2] + 100 ms
  TimeStep(0.1)@T[0]
    artiq/ttl1: Set(True)
  TimeStep(0.1)@T[2]
    artiq/ttl1: Pulse(False)
"""

    comp.finalize()

    assert str(s) == f"""Seq - T[2]
 T[0]: 0 ps
 T[1]: 100 ms
 T[2]: 200 ms
 T[3]: 0.3 s
  TimeStep(0.1)@T[0]
    artiq/ttl1: Set(True)
  TimeStep(0.1)@T[2]
    artiq/ttl1: Pulse(False)
"""

    assert test_utils.compiler_num_basic_seq(comp) == 1
    actions = test_utils.compiler_get_all_actions(comp, 0)
    assert len(actions) == 1
    assert len(actions[0]) == 2
    action1 = actions[0][0]
    action2 = actions[0][1]
    assert str(action1) == f'Set(True)'
    assert str(action2) == f'Pulse(False)'
    info1 = action1.get_compile_info()
    info2 = action2.get_compile_info()
    assert info1['tid'] == 0
    assert info1['end_tid'] == 1
    assert info1['length'] == 0.1
    assert info1['end_val'] is True

    assert info2['tid'] == 2
    assert info2['end_tid'] == 3
    assert info2['length'] == 0.1
    assert info2['end_val'] is True

    comp.runtime_finalize(1)
    assert action1.get_cond_val() is True
    assert action2.get_cond_val() is True
    total_time, times = test_utils.compiler_get_all_times(comp, 0)
    assert test_utils.compiler_get_channel_changed(comp) == [True]

    assert times == [0,
                     100_000_000_000,
                     200_000_000_000,
                     300_000_000_000]
    assert total_time == 300_000_000_000

    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.add_background(0.1).pulse('artiq/ttl1', True)
    s.wait(0.2)
    def jdaksf8923jfasd():
        s.add_background(0.1).set('artiq/ttl1', False)
    jdaksf8923jfasd()

    assert str(s) == f"""Seq - T[2]
 T[0]: 0 ps
 T[1]: T[0] + 100 ms
 T[2]: T[0] + 200 ms
 T[3]: T[2] + 100 ms
  TimeStep(0.1)@T[0]
    artiq/ttl1: Pulse(True)
  TimeStep(0.1)@T[2]
    artiq/ttl1: Set(False)
"""

    with pytest.raises(ValueError,
                       match="Actions on artiq/ttl1 is not statically ordered") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'jdaksf8923jfasd')

@test_utils.with_seq_params
def test_ramp_order_error1(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq

    f1 = test_utils.StaticFunction()
    s.add_background(0.1).set('artiq/analog', f1)
    s.wait(0.2)
    def fjasldf918327():
        s.add_background(0.1).pulse('artiq/analog', 0.1)
    fjasldf918327()

    assert str(s) == f"""Seq - T[2]
 T[0]: 0 ps
 T[1]: T[0] + 100 ms
 T[2]: T[0] + 200 ms
 T[3]: T[2] + 100 ms
  TimeStep(0.1)@T[0]
    artiq/analog: Set({f1})
  TimeStep(0.1)@T[2]
    artiq/analog: Pulse(0.1)
"""

    with pytest.raises(ValueError,
                       match="Actions on artiq/analog is not statically ordered") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'fjasldf918327')

@test_utils.with_seq_params
def test_ramp_order_error2(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq

    f1 = test_utils.StaticFunction()
    s.conditional(False).add_background(0.1).set('artiq/analog', f1)
    s.wait(0.2)
    def j98asdfjk2398asdf():
        s.conditional(False).add_background(0.1).pulse('artiq/analog', 0.1)
    j98asdfjk2398asdf()

    assert str(s) == f"""Seq - T[2]
 T[0]: 0 ps
 T[1]: T[0] + (100 ms; if False)
 T[2]: T[0] + 200 ms
 T[3]: T[2] + (100 ms; if False)
  TimeStep(0.1)@T[0] if False
    artiq/analog: Set({f1}, cond=False)
  TimeStep(0.1)@T[2] if False
    artiq/analog: Pulse(0.1, cond=False)
"""

    with pytest.raises(ValueError,
                       match="Actions on artiq/analog is not statically ordered") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'j98asdfjk2398asdf')

@test_utils.with_seq_params
def test_ramp_order1(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq

    c1val = 0.1
    c1 = test_utils.new_extern(lambda: c1val)

    c2val = 0.1
    c2 = test_utils.new_extern(lambda: c2val)

    s.set('artiq/analog', c1)
    f1 = test_utils.StaticFunction()
    step = s.add_background(c2).set('artiq/analog', f1)
    s.wait(0.2)
    s.wait_for(step)
    s.add_background(0.1).pulse('artiq/analog', 0.1)

    assert str(s) == f"""Seq - T[4]
 T[0]: 0 ps
 T[1]: T[0] + 0 ps
 T[2]: T[1] + int64({c2} * 1000000000000)
 T[3]: T[1] + 200 ms
 T[4]: T[3]; wait_for(T[2] + 0 ps)
 T[5]: T[4] + 100 ms
  TimeStep(0)@T[0]
    artiq/analog: Set({c1})
  TimeStep({c2})@T[1]
    artiq/analog: Set({f1})
  TimeStep(0.1)@T[4]
    artiq/analog: Pulse(0.1)
"""

    comp.finalize()

    assert test_utils.compiler_num_basic_seq(comp) == 1
    actions = test_utils.compiler_get_all_actions(comp, 0)
    assert len(actions) == 1
    assert len(actions[0]) == 3
    action1 = actions[0][0]
    action2 = actions[0][1]
    action3 = actions[0][2]
    assert str(action1) == f'Set({c1})'
    assert str(action2) == f'Set({f1})'
    assert str(action3) == f'Pulse(0.1)'
    info1 = action1.get_compile_info()
    info2 = action2.get_compile_info()
    info3 = action3.get_compile_info()
    assert info1['tid'] == 0
    assert info1['end_tid'] == 1
    assert info1['length'] == 0
    assert info1['end_val'] is c1

    assert info2['tid'] == 1
    assert info2['end_tid'] == 2
    assert info2['length'] is c2
    assert str(info2['end_val']) == f'({c2} / 2 + {c1}) - {c2}'

    assert info3['tid'] == 4
    assert info3['end_tid'] == 5
    assert info3['length'] == 0.1
    assert info3['end_val'] is info2['end_val']

    comp.runtime_finalize(1)
    assert action1.get_cond_val() is True
    assert action2.get_cond_val() is True
    assert action3.get_cond_val() is True
    total_time, times = test_utils.compiler_get_all_times(comp, 0)
    assert test_utils.compiler_get_channel_changed(comp) == [True]

    assert times == [0,
                     0,
                     100_000_000_000,
                     200_000_000_000,
                     200_000_000_000,
                     300_000_000_000]
    assert total_time == 300_000_000_000

    c2val = 0.5
    comp.runtime_finalize(2)
    assert action1.get_cond_val() is True
    assert action2.get_cond_val() is True
    assert action3.get_cond_val() is True
    total_time, times = test_utils.compiler_get_all_times(comp, 0)
    assert test_utils.compiler_get_channel_changed(comp) == [True]

    assert times == [0,
                     0,
                     500_000_000_000,
                     200_000_000_000,
                     500_000_000_000,
                     600_000_000_000]
    assert total_time == 600_000_000_000

@test_utils.with_seq_params
def test_rt_assert(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq

    c1val = True
    c1 = test_utils.new_extern(lambda: c1val)

    def japsidfjpaoisdjpafosd():
        s.rt_assert(c1, "Some message")
    japsidfjpaoisdjpafosd()

    comp.finalize()
    comp.runtime_finalize(1)
    assert test_utils.compiler_get_channel_changed(comp) == []

    c1val = False
    with pytest.raises(AssertionError, match="Some message") as exc:
        comp.runtime_finalize(2)
    test_utils.check_bt(exc, max_bt, 'japsidfjpaoisdjpafosd')

    c1val = True
    comp.runtime_finalize(3)
    assert test_utils.compiler_get_channel_changed(comp) == []

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    def assert_cb_JALKDFJOIADF():
        raise ValueError("XXX YYY ZZZ")
    c1 = test_utils.new_extern(assert_cb_JALKDFJOIADF)
    japsidfjpaoisdjpafosd()
    comp.finalize()
    with pytest.raises(ValueError, match="XXX YYY ZZZ") as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'japsidfjpaoisdjpafosd')
    test_utils.check_bt(exc, 1, 'assert_cb_JALKDFJOIADF')

@test_utils.with_seq_params
def test_cond_error(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.set('artiq/ttl0', True)
    s.add_step(0.01) \
      .pulse('artiq/urukul0_ch2/freq', test_utils.new_extern(lambda: 1.23)) \
      .pulse('artiq/ttl2', test_utils.new_extern(lambda: True))
    def error_callback():
        raise ValueError("AAABBBCCC")
    def ajqu7sdf7h7uhfasd():
        s.conditional(test_utils.new_extern(error_callback)) \
         .set('artiq/ttl0', True)
    ajqu7sdf7h7uhfasd()
    comp.finalize()
    with pytest.raises(ValueError, match="AAABBBCCC") as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'ajqu7sdf7h7uhfasd')

@test_utils.with_seq_params
def test_cond_ramp_error(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.conditional(test_utils.new_extern(lambda: False)) \
      .add_step(test_utils.new_extern(lambda: 0)) \
      .set('artiq/urukul0_ch0/amp', test_utils.DivLengthFunction()) \
      .pulse('artiq/urukul0_ch1/amp', test_utils.DivLengthFunction())
    comp.finalize()
    comp.runtime_finalize(1)
    assert test_utils.compiler_get_channel_changed(comp) == [True, True]

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.add_step(0) \
      .set('artiq/urukul0_ch0/amp', action.Blackman(1)) \
      .pulse('artiq/urukul0_ch1/amp', action.Blackman(1)) \
      .set('artiq/urukul1_ch0/amp', action.BlackmanSquare(1)) \
      .pulse('artiq/urukul1_ch1/amp', action.BlackmanSquare(1))
    comp.finalize()
    comp.runtime_finalize(1)
    assert test_utils.compiler_get_channel_changed(comp) == [True, True, True, True]

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.add_step(test_utils.new_extern(lambda: 0)) \
      .set('artiq/urukul0_ch0/amp', action.Blackman(1)) \
      .pulse('artiq/urukul0_ch1/amp', action.Blackman(1)) \
      .set('artiq/urukul1_ch0/amp', action.BlackmanSquare(1)) \
      .pulse('artiq/urukul1_ch1/amp', action.BlackmanSquare(1))
    comp.finalize()
    comp.runtime_finalize(1)
    assert test_utils.compiler_get_channel_changed(comp) == [True, True, True, True]

class MissingInit(action.RampFunction):
    def __init__(self):
        pass

    def eval(self, t, length, oldval):
        return t / length

@test_utils.with_seq_params
def test_ramp_noinit_error(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    def jaksdjfpoiasdnqeurfsda8u2jadf():
        s.add_step(2) \
          .set('artiq/urukul0_ch0/amp', MissingInit())
    jaksdjfpoiasdnqeurfsda8u2jadf()
    comp.finalize()
    with pytest.raises(RuntimeError, match="RampFunction.__init__ not called") as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'jaksdjfpoiasdnqeurfsda8u2jadf')

@test_utils.with_seq_params
def test_ramp_eval_end_error(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    def j8sjdfas():
        s.add_step(2) \
          .set('artiq/urukul0_ch0/amp', test_utils.ErrorEndFunction(ValueError("XYZ")))
    j8sjdfas()
    with pytest.raises(ValueError, match="XYZ") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'j8sjdfas')

@test_utils.with_seq_params
def test_invalid_flow(max_bt):
    def jlasdfjksdfkief48():
        return test_env.new_comp(max_bt)
    comp = jlasdfjksdfkief48()
    s = comp.seq
    s.add_branch(s)
    with pytest.raises(ValueError, match="Loop found in sequence") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'jlasdfjksdfkief48')

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    def jkiaj8f92fasd():
        s.new_basic_seq()
    jkiaj8f92fasd()
    with pytest.raises(ValueError, match="BasicSeq 1 unreachable") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'jkiaj8f92fasd')

@test_utils.with_seq_params
def test_bseq_empty(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    bt1 = s.new_basic_seq()
    bt2 = s.new_basic_seq()
    s.add_branch(bt1)
    bt1.add_branch(bt2)
    s.add_branch(bt2)
    comp.finalize()
    assert test_utils.compiler_num_basic_seq(comp) == 4

    assert test_utils.compiler_get_bseq_id(comp, 0) == 0
    assert test_utils.compiler_get_next_cbseq(comp, 0) == [1, 3]
    assert test_utils.compiler_get_all_start_values(comp, 0) == []
    assert test_utils.compiler_get_all_actions(comp, 0) == []

    assert test_utils.compiler_get_bseq_id(comp, 1) == 1
    assert test_utils.compiler_get_next_cbseq(comp, 1) == [2]
    assert test_utils.compiler_get_all_start_values(comp, 1) == []
    assert test_utils.compiler_get_all_actions(comp, 1) == []

    assert test_utils.compiler_get_bseq_id(comp, 2) == 2
    assert test_utils.compiler_get_next_cbseq(comp, 2) == [-1]
    assert test_utils.compiler_get_all_start_values(comp, 2) == []
    assert test_utils.compiler_get_all_actions(comp, 2) == []

    assert test_utils.compiler_get_bseq_id(comp, 3) == 2
    assert test_utils.compiler_get_next_cbseq(comp, 3) == [-1]
    assert test_utils.compiler_get_all_start_values(comp, 3) == []
    assert test_utils.compiler_get_all_actions(comp, 3) == []

    comp.runtime_finalize(1)
    assert test_utils.compiler_get_all_times(comp, 0) == (0, [0])
    assert test_utils.compiler_get_all_times(comp, 1) == (0, [0])
    assert test_utils.compiler_get_all_times(comp, 2) == (0, [0])
    assert test_utils.compiler_get_all_times(comp, 3) == (0, [0])
    assert test_utils.compiler_get_channel_changed(comp) == []

@test_utils.with_seq_params
def test_bseq_split(max_bt):
    r1 = test_utils.new_extern(lambda: 1)
    b1 = test_utils.new_extern(lambda: True)
    func_ramp = test_utils.FuncAction(lambda t, l, o: r1 * 2 - 0.4)
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.set('artiq/urukul0_ch0/amp', r1 * 3 + 2.0)
    bt1 = s.new_basic_seq()
    bt1.add_step(1).set('artiq/urukul0_ch0/amp', 1.0)
    bt2 = s.new_basic_seq()
    s.add_branch(bt1)
    bt1.add_branch(bt2)
    s.add_branch(bt2)
    comp.finalize()
    assert test_utils.compiler_num_basic_seq(comp) == 4

    assert test_utils.compiler_get_bseq_id(comp, 0) == 0
    assert test_utils.compiler_get_next_cbseq(comp, 0) == [1, 3]
    assert test_utils.compiler_get_all_start_values(comp, 0) == [0]
    actions0 = test_utils.compiler_get_all_actions(comp, 0)
    assert len(actions0) == 1
    assert len(actions0[0]) == 1
    assert str(actions0[0][0]) == f'Set(2.0 + {r1} * 3)'

    assert test_utils.compiler_get_bseq_id(comp, 1) == 1
    assert test_utils.compiler_get_next_cbseq(comp, 1) == [2]
    starts1 = test_utils.compiler_get_all_start_values(comp, 1)
    assert len(starts1) == 1
    assert str(starts1[0]) == f'2.0 + {r1} * 3'
    actions1 = test_utils.compiler_get_all_actions(comp, 1)
    assert len(actions1) == 1
    assert len(actions1[0]) == 1
    assert str(actions1[0][0]) == 'Set(1.0)'

    assert test_utils.compiler_get_bseq_id(comp, 2) == 2
    assert test_utils.compiler_get_next_cbseq(comp, 2) == [-1]
    assert test_utils.compiler_get_all_start_values(comp, 2) == [1.0]
    assert test_utils.compiler_get_all_actions(comp, 2) == [[]]

    assert test_utils.compiler_get_bseq_id(comp, 3) == 2
    assert test_utils.compiler_get_next_cbseq(comp, 3) == [-1]
    starts3 = test_utils.compiler_get_all_start_values(comp, 3)
    assert len(starts3) == 1
    assert starts3[0] is starts1[0]
    assert test_utils.compiler_get_all_actions(comp, 3) == [[]]
    assert not test_utils.compiler_check_action_reuse(comp, 2, 3, 0)

    comp.runtime_finalize(1)
    assert test_utils.compiler_get_all_times(comp, 0) == (0, [0, 0])
    assert test_utils.compiler_get_all_times(comp, 1) == (1000_000_000_000, [0, 1000_000_000_000])
    assert test_utils.compiler_get_all_times(comp, 2) == (0, [0])
    assert test_utils.compiler_get_all_times(comp, 3) == (0, [0])
    assert actions0[0][0].get_cond_val() is True
    assert actions1[0][0].get_cond_val() is True
    assert test_utils.compiler_get_channel_changed(comp) == [True]

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.set('artiq/urukul0_ch0/amp', r1 * 3 + 2.0)
    bt1 = s.new_basic_seq()
    bt1.add_step(1).set('artiq/urukul0_ch0/amp', 1.0)
    bt2 = s.new_basic_seq()
    s.add_branch(bt1)
    bt1.add_branch(bt2)
    s.add_branch(bt2)
    bt2.add_step(0.1).set('artiq/urukul0_ch0/amp', 0.2)
    comp.finalize()
    assert test_utils.compiler_num_basic_seq(comp) == 4

    assert test_utils.compiler_get_bseq_id(comp, 0) == 0
    assert test_utils.compiler_get_next_cbseq(comp, 0) == [1, 3]
    assert test_utils.compiler_get_all_start_values(comp, 0) == [0]
    actions0 = test_utils.compiler_get_all_actions(comp, 0)
    assert len(actions0) == 1
    assert len(actions0[0]) == 1
    assert str(actions0[0][0]) == f'Set(2.0 + {r1} * 3)'

    assert test_utils.compiler_get_bseq_id(comp, 1) == 1
    assert test_utils.compiler_get_next_cbseq(comp, 1) == [2]
    starts1 = test_utils.compiler_get_all_start_values(comp, 1)
    assert len(starts1) == 1
    assert str(starts1[0]) == f'2.0 + {r1} * 3'
    actions1 = test_utils.compiler_get_all_actions(comp, 1)
    assert len(actions1) == 1
    assert len(actions1[0]) == 1
    assert str(actions1[0][0]) == 'Set(1.0)'

    assert test_utils.compiler_get_bseq_id(comp, 2) == 2
    assert test_utils.compiler_get_next_cbseq(comp, 2) == [-1]
    assert test_utils.compiler_get_all_start_values(comp, 2) == [1.0]
    actions2 = test_utils.compiler_get_all_actions(comp, 2)
    assert len(actions2) == 1
    assert len(actions2[0]) == 1
    assert str(actions2[0][0]) == 'Set(0.2)'

    assert test_utils.compiler_get_bseq_id(comp, 3) == 2
    assert test_utils.compiler_get_next_cbseq(comp, 3) == [-1]
    starts3 = test_utils.compiler_get_all_start_values(comp, 3)
    assert len(starts3) == 1
    assert starts3[0] is starts1[0]
    actions3 = test_utils.compiler_get_all_actions(comp, 3)
    assert len(actions3) == 1
    assert len(actions3[0]) == 1
    assert str(actions3[0][0]) == 'Set(0.2)'
    assert not test_utils.compiler_check_action_reuse(comp, 2, 3, 0)

    comp.runtime_finalize(1)
    assert test_utils.compiler_get_all_times(comp, 0) == (0, [0, 0])
    assert test_utils.compiler_get_all_times(comp, 1) == (1000_000_000_000, [0, 1000_000_000_000])
    assert test_utils.compiler_get_all_times(comp, 2) == (100_000_000_000, [0, 100_000_000_000])
    assert test_utils.compiler_get_all_times(comp, 3) == (100_000_000_000, [0, 100_000_000_000])
    assert actions0[0][0].get_cond_val() is True
    assert actions1[0][0].get_cond_val() is True
    assert actions2[0][0].get_cond_val() is True
    assert actions3[0][0].get_cond_val() is True
    assert test_utils.compiler_get_channel_changed(comp) == [True]

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    v1 = r1 * 3 + 2.0
    s.set('artiq/urukul0_ch0/amp', v1)
    bt1 = s.new_basic_seq()
    bt1.add_step(1).set('artiq/urukul0_ch0/amp', 1.0)
    bt2 = s.new_basic_seq()
    s.add_branch(bt1)
    bt1.add_branch(bt2)
    s.add_branch(bt2)
    bt2.add_step(0.01).pulse('artiq/urukul0_ch0/amp', 0.2)
    comp.finalize()
    assert test_utils.compiler_num_basic_seq(comp) == 4

    assert test_utils.compiler_get_bseq_id(comp, 0) == 0
    assert test_utils.compiler_get_next_cbseq(comp, 0) == [1, 3]
    assert test_utils.compiler_get_all_start_values(comp, 0) == [0]
    actions0 = test_utils.compiler_get_all_actions(comp, 0)
    assert len(actions0) == 1
    assert len(actions0[0]) == 1
    assert str(actions0[0][0]) == f'Set(2.0 + {r1} * 3)'

    assert test_utils.compiler_get_bseq_id(comp, 1) == 1
    assert test_utils.compiler_get_next_cbseq(comp, 1) == [2]
    starts1 = test_utils.compiler_get_all_start_values(comp, 1)
    assert len(starts1) == 1
    assert str(starts1[0]) == f'2.0 + {r1} * 3'
    actions1 = test_utils.compiler_get_all_actions(comp, 1)
    assert len(actions1) == 1
    assert len(actions1[0]) == 1
    assert str(actions1[0][0]) == 'Set(1.0)'

    assert test_utils.compiler_get_bseq_id(comp, 2) == 2
    assert test_utils.compiler_get_next_cbseq(comp, 2) == [-1]
    assert test_utils.compiler_get_all_start_values(comp, 2) == [1.0]
    actions2 = test_utils.compiler_get_all_actions(comp, 2)
    assert len(actions2) == 1
    assert len(actions2[0]) == 1
    assert str(actions2[0][0]) == 'Pulse(0.2)'
    info2 = actions2[0][0].get_compile_info()
    assert info2['end_val'] == 1.0

    assert test_utils.compiler_get_bseq_id(comp, 3) == 2
    assert test_utils.compiler_get_next_cbseq(comp, 3) == [-1]
    starts3 = test_utils.compiler_get_all_start_values(comp, 3)
    assert len(starts3) == 1
    assert starts3[0] is starts1[0]
    actions3 = test_utils.compiler_get_all_actions(comp, 3)
    assert len(actions3) == 1
    assert len(actions3[0]) == 1
    assert str(actions3[0][0]) == 'Pulse(0.2)'
    info3 = actions3[0][0].get_compile_info()
    assert info3['end_val'] is v1
    assert not test_utils.compiler_check_action_reuse(comp, 2, 3, 0)

    comp.runtime_finalize(1)
    assert test_utils.compiler_get_all_times(comp, 0) == (0, [0, 0])
    assert test_utils.compiler_get_all_times(comp, 1) == (1000_000_000_000, [0, 1000_000_000_000])
    assert test_utils.compiler_get_all_times(comp, 2) == (10_000_000_000, [0, 10_000_000_000])
    assert test_utils.compiler_get_all_times(comp, 3) == (10_000_000_000, [0, 10_000_000_000])
    assert actions0[0][0].get_cond_val() is True
    assert actions1[0][0].get_cond_val() is True
    assert actions2[0][0].get_cond_val() is True
    assert actions3[0][0].get_cond_val() is True
    assert test_utils.compiler_get_channel_changed(comp) == [True]

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    v1 = r1 * 3 + 2.0
    s.set('artiq/urukul0_ch0/amp', v1)
    bt1 = s.new_basic_seq()
    bt1.add_step(1).set('artiq/urukul0_ch0/amp', r1 - 0.1)
    bt2 = s.new_basic_seq()
    s.add_branch(bt1)
    bt1.add_branch(bt2)
    s.add_branch(bt2)
    bt2.add_step(0.01).set('artiq/urukul0_ch0/amp', func_ramp)
    bt2.add_step(0.01).pulse('artiq/urukul0_ch0/amp', 1)
    comp.finalize()
    assert test_utils.compiler_num_basic_seq(comp) == 4

    assert test_utils.compiler_get_bseq_id(comp, 0) == 0
    assert test_utils.compiler_get_next_cbseq(comp, 0) == [1, 3]
    assert test_utils.compiler_get_all_start_values(comp, 0) == [0]
    actions0 = test_utils.compiler_get_all_actions(comp, 0)
    assert len(actions0) == 1
    assert len(actions0[0]) == 1
    assert str(actions0[0][0]) == f'Set(2.0 + {r1} * 3)'

    assert test_utils.compiler_get_bseq_id(comp, 1) == 1
    assert test_utils.compiler_get_next_cbseq(comp, 1) == [2]
    starts1 = test_utils.compiler_get_all_start_values(comp, 1)
    assert len(starts1) == 1
    assert str(starts1[0]) == f'2.0 + {r1} * 3'
    actions1 = test_utils.compiler_get_all_actions(comp, 1)
    assert len(actions1) == 1
    assert len(actions1[0]) == 1
    assert str(actions1[0][0]) == f'Set(-0.1 + {r1})'

    assert test_utils.compiler_get_bseq_id(comp, 2) == 2
    assert test_utils.compiler_get_next_cbseq(comp, 2) == [-1]
    starts2 = test_utils.compiler_get_all_start_values(comp, 2)
    assert len(starts2) == 1
    assert str(starts2[0]) == f'-0.1 + {r1}'
    actions2 = test_utils.compiler_get_all_actions(comp, 2)
    assert len(actions2) == 1
    assert len(actions2[0]) == 2
    assert str(actions2[0][0]) == f'Set({func_ramp})'
    assert str(actions2[0][1]) == 'Pulse(1)'
    info2 = actions2[0][1].get_compile_info()
    assert str(info2['end_val']) == f'-0.4 + {r1} * 2'

    assert test_utils.compiler_get_bseq_id(comp, 3) == 2
    assert test_utils.compiler_get_next_cbseq(comp, 3) == [-1]
    starts3 = test_utils.compiler_get_all_start_values(comp, 3)
    assert len(starts3) == 1
    assert starts3[0] is starts1[0]
    actions3 = test_utils.compiler_get_all_actions(comp, 3)
    assert len(actions3) == 1
    assert len(actions3[0]) == 2
    assert str(actions3[0][0]) == f'Set({func_ramp})'
    assert str(actions3[0][1]) == 'Pulse(1)'
    info3 = actions3[0][1].get_compile_info()
    assert info3['end_val'] is info2['end_val']
    assert not test_utils.compiler_check_action_reuse(comp, 2, 3, 0)

    comp.runtime_finalize(1)
    assert test_utils.compiler_get_all_times(comp, 0) == (0, [0, 0])
    assert test_utils.compiler_get_all_times(comp, 1) == (1000_000_000_000, [0, 1000_000_000_000])
    assert test_utils.compiler_get_all_times(comp, 2) == (20_000_000_000, [0, 10_000_000_000, 20_000_000_000])
    assert test_utils.compiler_get_all_times(comp, 3) == (20_000_000_000, [0, 10_000_000_000, 20_000_000_000])
    assert actions0[0][0].get_cond_val() is True
    assert actions1[0][0].get_cond_val() is True
    assert actions2[0][0].get_cond_val() is True
    assert actions2[0][1].get_cond_val() is True
    assert actions3[0][0].get_cond_val() is True
    assert actions3[0][1].get_cond_val() is True
    assert test_utils.compiler_get_channel_changed(comp) == [True]

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    v1 = r1 * 3 + 2.0
    s.set('artiq/urukul0_ch0/amp', v1)
    bt1 = s.new_basic_seq()
    bt1.add_step(1).set('artiq/urukul0_ch0/amp', r1 - 0.1)
    bt1.set('artiq/urukul0_ch0/amp', r1 * 3 + 2.0)
    bt2 = s.new_basic_seq()
    s.add_branch(bt1)
    bt1.add_branch(bt2)
    s.add_branch(bt2)
    bt2.add_step(0.01).set('artiq/urukul0_ch0/amp', func_ramp)
    bt2.add_step(0.01).set('artiq/urukul0_ch0/amp', 0.2)
    comp.finalize()
    assert test_utils.compiler_num_basic_seq(comp) == 4

    assert test_utils.compiler_get_bseq_id(comp, 0) == 0
    assert test_utils.compiler_get_next_cbseq(comp, 0) == [1, 3]
    assert test_utils.compiler_get_all_start_values(comp, 0) == [0]
    actions0 = test_utils.compiler_get_all_actions(comp, 0)
    assert len(actions0) == 1
    assert len(actions0[0]) == 1
    assert str(actions0[0][0]) == f'Set(2.0 + {r1} * 3)'

    assert test_utils.compiler_get_bseq_id(comp, 1) == 1
    assert test_utils.compiler_get_next_cbseq(comp, 1) == [2]
    starts1 = test_utils.compiler_get_all_start_values(comp, 1)
    assert len(starts1) == 1
    assert str(starts1[0]) == f'2.0 + {r1} * 3'
    actions1 = test_utils.compiler_get_all_actions(comp, 1)
    assert len(actions1) == 1
    assert len(actions1[0]) == 2
    assert str(actions1[0][0]) == f'Set(-0.1 + {r1})'
    assert str(actions1[0][1]) == f'Set(2.0 + {r1} * 3)'
    info10 = actions1[0][0].get_compile_info()
    assert str(info10['end_val']) == f'-0.1 + {r1}'
    info11 = actions1[0][1].get_compile_info()
    assert info11['end_val'] is not v1
    assert str(info11['end_val']) == str(v1)

    assert test_utils.compiler_get_bseq_id(comp, 2) == 2
    assert test_utils.compiler_get_next_cbseq(comp, 2) == [-1]
    starts2 = test_utils.compiler_get_all_start_values(comp, 2)
    assert len(starts2) == 1
    assert starts2[0] is info11['end_val']
    actions2 = test_utils.compiler_get_all_actions(comp, 2)
    assert len(actions2) == 1
    assert len(actions2[0]) == 2
    assert str(actions2[0][0]) == f'Set({func_ramp})'
    assert str(actions2[0][1]) == 'Set(0.2)'
    info20 = actions2[0][0].get_compile_info()
    assert str(info20['end_val']) == f'-0.4 + {r1} * 2'
    info21 = actions2[0][1].get_compile_info()
    assert info21['end_val'] == 0.2

    assert test_utils.compiler_get_bseq_id(comp, 3) == 2
    assert test_utils.compiler_get_next_cbseq(comp, 3) == [-1]
    starts3 = test_utils.compiler_get_all_start_values(comp, 3)
    assert len(starts3) == 1
    assert starts3[0] is info11['end_val']
    actions3 = test_utils.compiler_get_all_actions(comp, 3)
    assert len(actions3) == 1
    assert len(actions3[0]) == 2
    assert str(actions3[0][0]) == f'Set({func_ramp})'
    assert str(actions3[0][1]) == 'Set(0.2)'
    info30 = actions3[0][0].get_compile_info()
    assert info30['end_val'] is info20['end_val']
    info31 = actions3[0][1].get_compile_info()
    assert info31['end_val'] is info21['end_val']
    assert test_utils.compiler_check_action_reuse(comp, 2, 3, 0)

    comp.runtime_finalize(1)
    assert test_utils.compiler_get_all_times(comp, 0) == (0, [0, 0])
    assert test_utils.compiler_get_all_times(comp, 1) == (1000_000_000_000, [0, 1000_000_000_000, 1000_000_000_000])
    assert test_utils.compiler_get_all_times(comp, 2) == (20_000_000_000, [0, 10_000_000_000, 20_000_000_000])
    assert test_utils.compiler_get_all_times(comp, 3) == (20_000_000_000, [0, 10_000_000_000, 20_000_000_000])
    assert actions0[0][0].get_cond_val() is True
    assert actions1[0][0].get_cond_val() is True
    assert actions2[0][0].get_cond_val() is True
    assert actions2[0][1].get_cond_val() is True
    assert actions3[0][0].get_cond_val() is True
    assert actions3[0][1].get_cond_val() is True
    assert test_utils.compiler_get_channel_changed(comp) == [True]

@test_utils.with_seq_params
def test_channel_change(max_bt):
    t0 = rtval.seq_variable(0.1)
    t1 = rtval.seq_variable(0.1)
    t2 = rtval.seq_variable(0.1)
    v1 = rtval.seq_variable(0.1)
    v2 = rtval.seq_variable(0.1)
    v3 = rtval.seq_variable(0.1)
    v4 = rtval.seq_variable(0.1)

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    @s.add_background
    def f(s):
        s.wait(t0)
        s.add_step(t1) \
          .set('artiq/urukul0_ch0/amp', action.Blackman(v1)) \
          .pulse('artiq/urukul0_ch1/amp', action.Blackman(v2))
        s.add_step(t2) \
          .set('artiq/urukul1_ch0/amp', v3) \
          .pulse('artiq/urukul1_ch1/amp', v4)
    s.wait(2) # Allow changing t1 and t2 without changing total sequence length
    comp.finalize()

    comp.runtime_finalize(1)
    assert test_utils.compiler_get_channel_changed(comp) == [True, True, True, True]

    comp.runtime_finalize(2)
    assert test_utils.compiler_get_channel_changed(comp) == [False, False, False, False]

    t0.value = 0.2
    comp.runtime_finalize(3)
    assert test_utils.compiler_get_channel_changed(comp) == [True, True, True, True]

    t0.value = 0.1
    t1.value = 0.2
    comp.runtime_finalize(4)
    assert test_utils.compiler_get_channel_changed(comp) == [True, True, False, False]

    t1.value = 0.1
    comp.runtime_finalize(5)
    assert test_utils.compiler_get_channel_changed(comp) == [True, True, True, True]

    t2.value = 0.2
    comp.runtime_finalize(6)
    assert test_utils.compiler_get_channel_changed(comp) == [False, False, False, True]

    # Total time change invalidates all channels
    t2.value = 3
    comp.runtime_finalize(7)
    assert test_utils.compiler_get_channel_changed(comp) == [True, True, True, True]

    v1.value = 0.2
    comp.runtime_finalize(8)
    assert test_utils.compiler_get_channel_changed(comp) == [True, False, False, False]

    v2.value = 0.2
    comp.runtime_finalize(9)
    assert test_utils.compiler_get_channel_changed(comp) == [False, True, False, False]

    # Plain value change tracking is up to the backend
    v3.value = 0.2
    comp.runtime_finalize(10)
    assert test_utils.compiler_get_channel_changed(comp) == [False, False, False, False]

    # Plain value change tracking is up to the backend
    v4.value = 0.2
    comp.runtime_finalize(11)
    assert test_utils.compiler_get_channel_changed(comp) == [False, False, False, False]

    comp.runtime_finalize(12)
    assert test_utils.compiler_get_channel_changed(comp) == [False, False, False, False]

    comp.force_recompile = True
    comp.runtime_finalize(13)
    assert test_utils.compiler_get_channel_changed(comp) == [True, True, True, True]
