#

from brassboard_seq.config import Config
from brassboard_seq import seq, rtval, action
import brassboard_seq_test_utils as test_utils
import pytest

global_conf = Config()
global_conf.add_supported_prefix('artiq')

def new_seq():
    return seq.new_seq(global_conf)

def test_cond_order1():
    s = new_seq()

    c1val = True
    c1 = rtval.new_extern(lambda: c1val)
    c2val = True
    c2 = rtval.new_extern(lambda: c2val)

    def step1(s):
        s.wait(0.1)
        s.conditional(c1).add_step(1).set('artiq/ttl1', True)
    st1 = s.add_background(step1)
    def step2(s):
        s.conditional(c2).wait_for(st1)
        s.add_step(2).set('artiq/ttl1', False)
    st2 = s.add_step(step2)

    assert str(s) == f"""Seq - T[4]
 T[0]: 0.0 ps
 T[1]: T[0] + 100.0 ms
 T[2]: T[1] + (1.0 s; if bool({c1}))
 T[3]: T[0]; wait_for(T[2] + 0.0 ps; if bool({c2}))
 T[4]: T[3] + 2.0 s
  SubSeq@T[0] - T[2]
    TimeStep(1)@T[1] if bool({c1})
      artiq/ttl1: Set(True, cond=bool({c1}))
  SubSeq@T[0] - T[4]
    TimeStep(2)@T[3]
      artiq/ttl1: Set(False)
"""

    test_utils.seq_finalize(s)

    assert str(s) == f"""Seq - T[4]
 T[0]: 0.0 ps
 T[1]: T[0] + 100.0 ms
 T[2]: T[1] + (1.0 s; if bool({c1}))
 T[3]: T[0]; wait_for(T[2] + 0.0 ps; if bool({c2}))
 T[4]: T[3] + 2.0 s
  SubSeq@T[0] - T[2]
    TimeStep(1)@T[1] if bool({c1})
      artiq/ttl1: Set(True, cond=bool({c1}))
  SubSeq@T[0] - T[4]
    TimeStep(2)@T[3]
      artiq/ttl1: Set(False)
"""

    actions = test_utils.seq_get_all_actions(s)
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
    assert info1['prev_val'] == 0
    assert info1['length'] == 1
    assert str(info1['end_val']) == f'ifelse(bool({c1}), True, 0)'

    assert info2['tid'] == 3
    assert info2['end_tid'] == 4
    assert info2['prev_val'] is info1['end_val']
    assert info2['length'] == 2
    assert info2['end_val'] is False

def test_cond_order2():
    s = new_seq()

    c1val = True
    c1 = rtval.new_extern(lambda: c1val)
    c2val = True
    c2 = rtval.new_extern(lambda: c2val)

    st1 = s.add_floating(2).set('artiq/ttl1', False)
    def step2(s):
        s.wait(0.1)
        s.conditional(c1).add_step(1).set('artiq/ttl1', True)
    st2 = s.add_background(step2)
    s.conditional(c2).wait_for(st2)
    st1.set_time(s.current_time)

    assert str(s) == f"""Seq - T[5]
 T[0]: 0.0 ps
 T[1]: T[5] + 0.0 ps
 T[2]: T[1] + 2.0 s
 T[3]: T[0] + 100.0 ms
 T[4]: T[3] + (1.0 s; if bool({c1}))
 T[5]: T[0]; wait_for(T[4] + 0.0 ps; if bool({c2}))
  TimeStep(2)@T[1]
    artiq/ttl1: Set(False)
  SubSeq@T[0] - T[4]
    TimeStep(1)@T[3] if bool({c1})
      artiq/ttl1: Set(True, cond=bool({c1}))
"""

    test_utils.seq_finalize(s)

    assert str(s) == f"""Seq - T[3]
 T[0]: 0.0 ps
 T[1]: T[0] + 100.0 ms
 T[2]: T[1] + (1.0 s; if bool({c1}))
 T[3]: T[0]; wait_for(T[2] + 0.0 ps; if bool({c2}))
 T[4]: T[3] + 0.0 ps
 T[5]: T[4] + 2.0 s
  TimeStep(2)@T[4]
    artiq/ttl1: Set(False)
  SubSeq@T[0] - T[2]
    TimeStep(1)@T[1] if bool({c1})
      artiq/ttl1: Set(True, cond=bool({c1}))
"""

    actions = test_utils.seq_get_all_actions(s)
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
    assert info1['prev_val'] == 0
    assert info1['length'] == 1
    assert str(info1['end_val']) == f'ifelse(bool({c1}), True, 0)'

    assert info2['tid'] == 4
    assert info2['end_tid'] == 5
    assert info2['prev_val'] is info1['end_val']
    assert info2['length'] == 2
    assert info2['end_val'] is False

def test_order_error1():
    s = new_seq()

    c1val = True
    c1 = rtval.new_extern(lambda: c1val)

    st1 = s.conditional(c1).add_background(1).set('artiq/ttl1', True)
    st2 = s.conditional(rtval.inv(c1)).add_background(2).set('artiq/ttl1', True)

    assert str(s) == f"""Seq - T[0]
 T[0]: 0.0 ps
 T[1]: T[0] + (1.0 s; if bool({c1}))
 T[2]: T[0] + (2.0 s; if bool(inv({c1})))
  TimeStep(1)@T[0] if bool({c1})
    artiq/ttl1: Set(True, cond=bool({c1}))
  TimeStep(2)@T[0] if bool(inv({c1}))
    artiq/ttl1: Set(True, cond=bool(inv({c1})))
"""

    with pytest.raises(ValueError,
                       match="Actions on artiq/ttl1 is not statically ordered"):
        test_utils.seq_finalize(s)

def test_order_error2():
    s = new_seq()

    c1val = True
    c1 = rtval.new_extern(lambda: c1val)

    def step(s):
        s.wait(0.1)
        s.set('artiq/ttl1', True)
    st1 = s.conditional(c1).add_background(step)
    st2 = s.conditional(rtval.inv(c1)).add_background(step)

    assert str(s) == f"""Seq - T[0]
 T[0]: 0.0 ps
 T[1]: T[0] + (100.0 ms; if bool({c1}))
 T[2]: T[1] + (0.0 ps; if bool({c1}))
 T[3]: T[0] + (100.0 ms; if bool(inv({c1})))
 T[4]: T[3] + (0.0 ps; if bool(inv({c1})))
  SubSeq@T[0] - T[2] if bool({c1})
    TimeStep(0)@T[1] if bool({c1})
      artiq/ttl1: Set(True, cond=bool({c1}))
  SubSeq@T[0] - T[4] if bool(inv({c1}))
    TimeStep(0)@T[3] if bool(inv({c1}))
      artiq/ttl1: Set(True, cond=bool(inv({c1})))
"""

    with pytest.raises(ValueError,
                       match="Actions on artiq/ttl1 is not statically ordered"):
        test_utils.seq_finalize(s)

def test_order_error3():
    s = new_seq()

    s.add_background(0.1).set('artiq/ttl1', True)
    s.wait(0.2)
    s.add_background(0.1).set('artiq/ttl1', False)

    assert str(s) == f"""Seq - T[2]
 T[0]: 0.0 ps
 T[1]: T[0] + 100.0 ms
 T[2]: T[0] + 200.0 ms
 T[3]: T[2] + 100.0 ms
  TimeStep(0.1)@T[0]
    artiq/ttl1: Set(True)
  TimeStep(0.1)@T[2]
    artiq/ttl1: Set(False)
"""

    test_utils.seq_finalize(s)

    assert str(s) == f"""Seq - T[2]
 T[0]: 0.0 ps
 T[1]: T[0] + 100.0 ms
 T[2]: T[0] + 200.0 ms
 T[3]: T[2] + 100.0 ms
  TimeStep(0.1)@T[0]
    artiq/ttl1: Set(True)
  TimeStep(0.1)@T[2]
    artiq/ttl1: Set(False)
"""

    actions = test_utils.seq_get_all_actions(s)
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
    assert info1['prev_val'] == 0
    assert info1['length'] == 0.1
    assert info1['end_val'] is True

    assert info2['tid'] == 2
    assert info2['end_tid'] == 3
    assert info2['prev_val'] is True
    assert info2['length'] == 0.1
    assert info2['end_val'] is False

    s = new_seq()

    s.add_background(0.1).set('artiq/ttl1', True)
    s.wait(0.2)
    s.add_background(0.1).pulse('artiq/ttl1', False)

    assert str(s) == f"""Seq - T[2]
 T[0]: 0.0 ps
 T[1]: T[0] + 100.0 ms
 T[2]: T[0] + 200.0 ms
 T[3]: T[2] + 100.0 ms
  TimeStep(0.1)@T[0]
    artiq/ttl1: Set(True)
  TimeStep(0.1)@T[2]
    artiq/ttl1: Pulse(False)
"""

    test_utils.seq_finalize(s)

    assert str(s) == f"""Seq - T[2]
 T[0]: 0.0 ps
 T[1]: T[0] + 100.0 ms
 T[2]: T[0] + 200.0 ms
 T[3]: T[2] + 100.0 ms
  TimeStep(0.1)@T[0]
    artiq/ttl1: Set(True)
  TimeStep(0.1)@T[2]
    artiq/ttl1: Pulse(False)
"""

    actions = test_utils.seq_get_all_actions(s)
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
    assert info1['prev_val'] == 0
    assert info1['length'] == 0.1
    assert info1['end_val'] is True

    assert info2['tid'] == 2
    assert info2['end_tid'] == 3
    assert info2['prev_val'] is True
    assert info2['length'] == 0.1
    assert info2['end_val'] is True

    s = new_seq()

    s.add_background(0.1).pulse('artiq/ttl1', True)
    s.wait(0.2)
    s.add_background(0.1).set('artiq/ttl1', False)

    assert str(s) == f"""Seq - T[2]
 T[0]: 0.0 ps
 T[1]: T[0] + 100.0 ms
 T[2]: T[0] + 200.0 ms
 T[3]: T[2] + 100.0 ms
  TimeStep(0.1)@T[0]
    artiq/ttl1: Pulse(True)
  TimeStep(0.1)@T[2]
    artiq/ttl1: Set(False)
"""

    with pytest.raises(ValueError,
                       match="Actions on artiq/ttl1 is not statically ordered"):
        test_utils.seq_finalize(s)

class StaticFunction(action.RampFunction):
    def __init__(self):
        action.RampFunction.__init__(self)

    def eval(self, t, length, oldval):
        return t / 2 + oldval - length

def test_ramp_order_error1():
    s = new_seq()

    f1 = StaticFunction()
    s.add_background(0.1).set('artiq/analog', f1)
    s.wait(0.2)
    s.add_background(0.1).pulse('artiq/analog', 0.1)

    assert str(s) == f"""Seq - T[2]
 T[0]: 0.0 ps
 T[1]: T[0] + 100.0 ms
 T[2]: T[0] + 200.0 ms
 T[3]: T[2] + 100.0 ms
  TimeStep(0.1)@T[0]
    artiq/analog: Set({f1})
  TimeStep(0.1)@T[2]
    artiq/analog: Pulse(0.1)
"""

    with pytest.raises(ValueError,
                       match="Actions on artiq/analog is not statically ordered"):
        test_utils.seq_finalize(s)

def test_ramp_order1():
    s = new_seq()

    c1val = 0.1
    c1 = rtval.new_extern(lambda: c1val)

    c2val = 0.1
    c2 = rtval.new_extern(lambda: c2val)

    s.set('artiq/analog', c1)
    f1 = StaticFunction()
    step = s.add_background(c2).set('artiq/analog', f1)
    s.wait(0.2)
    s.wait_for(step)
    s.add_background(0.1).pulse('artiq/analog', 0.1)

    assert str(s) == f"""Seq - T[4]
 T[0]: 0.0 ps
 T[1]: T[0] + 0.0 ps
 T[2]: T[1] + int64({c2} * 1000000000000)
 T[3]: T[1] + 200.0 ms
 T[4]: T[3]; wait_for(T[2] + 0.0 ps)
 T[5]: T[4] + 100.0 ms
  TimeStep(0)@T[0]
    artiq/analog: Set({c1})
  TimeStep({c2})@T[1]
    artiq/analog: Set({f1})
  TimeStep(0.1)@T[4]
    artiq/analog: Pulse(0.1)
"""

    test_utils.seq_finalize(s)

    actions = test_utils.seq_get_all_actions(s)
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
    assert info1['prev_val'] == 0
    assert info1['length'] == 0
    assert info1['end_val'] is c1

    assert info2['tid'] == 1
    assert info2['end_tid'] == 2
    assert info2['prev_val'] is c1
    assert info2['length'] is c2
    assert str(info2['end_val']) == f'({c2} / 2 + {c1}) - {c2}'

    assert info3['tid'] == 4
    assert info3['end_tid'] == 5
    assert info3['prev_val'] is info2['end_val']
    assert info3['length'] == 0.1
    assert info3['end_val'] is info2['end_val']
