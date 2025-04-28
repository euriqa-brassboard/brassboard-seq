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

    actions = test_utils.compiler_get_all_actions(comp)
    assert len(actions) == 1
    assert len(actions[0]) == 2
    action1 = actions[0][0]
    action2 = actions[0][1]
    assert str(action1) == f'Set(True, cond=bool({c1}))'
    assert str(action2) == f'Set(False)'
    info1 = test_utils.action_get_compile_info(action1)
    info2 = test_utils.action_get_compile_info(action2)
    assert info1['tid'] == 1
    assert info1['end_tid'] == 2
    assert info1['length'] == 1
    assert str(info1['end_val']) == f'ifelse(bool({c1}), True, False)'

    assert info2['tid'] == 3
    assert info2['end_tid'] == 4
    assert info2['length'] == 2
    assert info2['end_val'] is False

    comp.runtime_finalize(1)
    assert test_utils.action_get_cond_val(action1) is True
    assert test_utils.action_get_cond_val(action2) is True
    total_time, times = test_utils.compiler_get_all_times(comp)

    assert times == [0,
                     100_000_000_000,
                     1100_000_000_000,
                     1100_000_000_000,
                     3100_000_000_000]
    assert total_time == 3100_000_000_000

    c1val = False
    comp.runtime_finalize(2)
    assert test_utils.action_get_cond_val(action1) is c1val
    assert test_utils.action_get_cond_val(action2) is True
    total_time, times = test_utils.compiler_get_all_times(comp)

    assert times == [0,
                     100_000_000_000,
                     100_000_000_000,
                     100_000_000_000,
                     2100_000_000_000]
    assert total_time == 2100_000_000_000

    c2val = False
    comp.runtime_finalize(3)
    assert test_utils.action_get_cond_val(action1) is c1val
    assert test_utils.action_get_cond_val(action2) is True
    total_time, times = test_utils.compiler_get_all_times(comp)

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

    actions = test_utils.compiler_get_all_actions(comp)
    assert len(actions) == 1
    assert len(actions[0]) == 2
    action1 = actions[0][0]
    action2 = actions[0][1]
    assert str(action1) == f'Set(True, cond=bool({c1}))'
    assert str(action2) == f'Set(False)'
    info1 = test_utils.action_get_compile_info(action1)
    info2 = test_utils.action_get_compile_info(action2)
    assert info1['tid'] == 1
    assert info1['end_tid'] == 2
    assert info1['length'] == 1
    assert str(info1['end_val']) == f'ifelse(bool({c1}), True, False)'

    assert info2['tid'] == 4
    assert info2['end_tid'] == 5
    assert info2['length'] == 2
    assert info2['end_val'] is False

    comp.runtime_finalize(1)
    assert test_utils.action_get_cond_val(action1) is True
    assert test_utils.action_get_cond_val(action2) is True
    total_time, times = test_utils.compiler_get_all_times(comp)

    assert times == [0,
                     100_000_000_000,
                     1100_000_000_000,
                     1100_000_000_000,
                     1100_000_000_000,
                     3100_000_000_000]
    assert total_time == 3100_000_000_000

    c1val = False
    comp.runtime_finalize(2)
    assert test_utils.action_get_cond_val(action1) is c1val
    assert test_utils.action_get_cond_val(action2) is True
    total_time, times = test_utils.compiler_get_all_times(comp)

    assert times == [0,
                     100_000_000_000,
                     100_000_000_000,
                     100_000_000_000,
                     100_000_000_000,
                     2100_000_000_000]
    assert total_time == 2100_000_000_000

    c2val = False
    comp.runtime_finalize(3)
    assert test_utils.action_get_cond_val(action1) is c1val
    assert test_utils.action_get_cond_val(action2) is True
    total_time, times = test_utils.compiler_get_all_times(comp)

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

    actions = test_utils.compiler_get_all_actions(comp)
    assert len(actions) == 1
    assert len(actions[0]) == 2
    action1 = actions[0][0]
    action2 = actions[0][1]
    assert str(action1) == f'Set(True)'
    assert str(action2) == f'Set(False)'
    info1 = test_utils.action_get_compile_info(action1)
    info2 = test_utils.action_get_compile_info(action2)
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

    actions = test_utils.compiler_get_all_actions(comp)
    assert len(actions) == 1
    assert len(actions[0]) == 2
    action1 = actions[0][0]
    action2 = actions[0][1]
    assert str(action1) == f'Set(True)'
    assert str(action2) == f'Pulse(False)'
    info1 = test_utils.action_get_compile_info(action1)
    info2 = test_utils.action_get_compile_info(action2)
    assert info1['tid'] == 0
    assert info1['end_tid'] == 1
    assert info1['length'] == 0.1
    assert info1['end_val'] is True

    assert info2['tid'] == 2
    assert info2['end_tid'] == 3
    assert info2['length'] == 0.1
    assert info2['end_val'] is True

    comp.runtime_finalize(1)
    assert test_utils.action_get_cond_val(action1) is True
    assert test_utils.action_get_cond_val(action2) is True
    total_time, times = test_utils.compiler_get_all_times(comp)

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

    actions = test_utils.compiler_get_all_actions(comp)
    assert len(actions) == 1
    assert len(actions[0]) == 3
    action1 = actions[0][0]
    action2 = actions[0][1]
    action3 = actions[0][2]
    assert str(action1) == f'Set({c1})'
    assert str(action2) == f'Set({f1})'
    assert str(action3) == f'Pulse(0.1)'
    info1 = test_utils.action_get_compile_info(action1)
    info2 = test_utils.action_get_compile_info(action2)
    info3 = test_utils.action_get_compile_info(action3)
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
    assert test_utils.action_get_cond_val(action1) is True
    assert test_utils.action_get_cond_val(action2) is True
    assert test_utils.action_get_cond_val(action3) is True
    total_time, times = test_utils.compiler_get_all_times(comp)

    assert times == [0,
                     0,
                     100_000_000_000,
                     200_000_000_000,
                     200_000_000_000,
                     300_000_000_000]
    assert total_time == 300_000_000_000

    c2val = 0.5
    comp.runtime_finalize(2)
    assert test_utils.action_get_cond_val(action1) is True
    assert test_utils.action_get_cond_val(action2) is True
    assert test_utils.action_get_cond_val(action3) is True
    total_time, times = test_utils.compiler_get_all_times(comp)

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

    c1val = False
    with pytest.raises(AssertionError, match="Some message") as exc:
        comp.runtime_finalize(2)
    test_utils.check_bt(exc, max_bt, 'japsidfjpaoisdjpafosd')

    c1val = True
    comp.runtime_finalize(3)

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

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.add_step(0) \
      .set('artiq/urukul0_ch0/amp', action.Blackman(1)) \
      .pulse('artiq/urukul0_ch1/amp', action.Blackman(1)) \
      .set('artiq/urukul1_ch0/amp', action.BlackmanSquare(1)) \
      .pulse('artiq/urukul1_ch1/amp', action.BlackmanSquare(1))
    comp.finalize()
    comp.runtime_finalize(1)

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.add_step(test_utils.new_extern(lambda: 0)) \
      .set('artiq/urukul0_ch0/amp', action.Blackman(1)) \
      .pulse('artiq/urukul0_ch1/amp', action.Blackman(1)) \
      .set('artiq/urukul1_ch0/amp', action.BlackmanSquare(1)) \
      .pulse('artiq/urukul1_ch1/amp', action.BlackmanSquare(1))
    comp.finalize()
    comp.runtime_finalize(1)

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
