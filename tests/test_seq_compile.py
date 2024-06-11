#

from brassboard_seq.config import Config
from brassboard_seq import seq, rtval, action
import brassboard_seq_test_utils as test_utils
import pytest

global_conf = Config()
global_conf.add_supported_prefix('artiq')

def new_seq(*args):
    return seq.Seq(global_conf, *args)

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

with_seq_params = with_params((0,), (5,), (500,))

@with_seq_params
def test_cond_order1(max_bt):
    s = new_seq(max_bt)

    c1val = True
    c1 = rtval.new_extern(lambda: c1val)
    c2val = True
    c2 = rtval.new_extern(lambda: c2val)

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

    test_utils.seq_finalize(s)

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

@with_seq_params
def test_cond_order2(max_bt):
    s = new_seq(max_bt)

    c1val = True
    c1 = rtval.new_extern(lambda: c1val)
    c2val = True
    c2 = rtval.new_extern(lambda: c2val)

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

    test_utils.seq_finalize(s)

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

@with_seq_params
def test_order_error1(max_bt):
    s = new_seq(max_bt)

    c1val = True
    c1 = rtval.new_extern(lambda: c1val)

    def f1_89374asjkl_asbd():
        st1 = s.conditional(c1).add_background(1).set('artiq/ttl1', True)
        st2 = s.conditional(rtval.inv(c1)).add_background(2).set('artiq/ttl1', True)
    def gj18j_9034rfj():
        f1_89374asjkl_asbd()
    gj18j_9034rfj()

    assert str(s) == f"""Seq - T[0]
 T[0]: 0 ps
 T[1]: T[0] + (1 s; if bool({c1}))
 T[2]: T[0] + (2 s; if bool(inv({c1})))
  TimeStep(1)@T[0] if bool({c1})
    artiq/ttl1: Set(True, cond=bool({c1}))
  TimeStep(2)@T[0] if bool(inv({c1}))
    artiq/ttl1: Set(True, cond=bool(inv({c1})))
"""

    with pytest.raises(ValueError,
                       match="Multiple actions added for the same channel "
                       "at the same time on artiq/ttl1") as exc:
        test_utils.seq_finalize(s)
    check_bt(exc, max_bt, 'f1_89374asjkl_asbd', 'gj18j_9034rfj')

@with_seq_params
def test_order_error2(max_bt):
    s = new_seq(max_bt)

    c1val = True
    c1 = rtval.new_extern(lambda: c1val)

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
 T[3]: T[0] + (100 ms; if bool(inv({c1})))
 T[4]: T[3] + (0 ps; if bool(inv({c1})))
  SubSeq@T[0] - T[2] if bool({c1})
    TimeStep(0)@T[1] if bool({c1})
      artiq/ttl1: Set(True, cond=bool({c1}))
  SubSeq@T[0] - T[4] if bool(inv({c1}))
    TimeStep(0)@T[3] if bool(inv({c1}))
      artiq/ttl1: Set(True, cond=bool(inv({c1})))
"""

    with pytest.raises(ValueError,
                       match="Actions on artiq/ttl1 is not statically ordered") as exc:
        test_utils.seq_finalize(s)
    check_bt(exc, max_bt, 'f_1239', 'step_13238324', 'kjaksd9123')

@with_seq_params
def test_order_error3(max_bt):
    s = new_seq(max_bt)

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

    test_utils.seq_finalize(s)

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

    s = new_seq(max_bt)

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

    test_utils.seq_finalize(s)

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

    s = new_seq(max_bt)

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
        test_utils.seq_finalize(s)
    check_bt(exc, max_bt, 'jdaksf8923jfasd')

class StaticFunction(action.RampFunction):
    def __init__(self):
        action.RampFunction.__init__(self)

    def eval(self, t, length, oldval):
        return t / 2 + oldval - length

@with_seq_params
def test_ramp_order_error1(max_bt):
    s = new_seq(max_bt)

    f1 = StaticFunction()
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
        test_utils.seq_finalize(s)
    check_bt(exc, max_bt, 'fjasldf918327')

@with_seq_params
def test_ramp_order1(max_bt):
    s = new_seq(max_bt)

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

class ErrorFunction(action.RampFunction):
    def __init__(self, err):
        action.RampFunction.__init__(self)
        self.err = err

    def eval(self, t, length, oldval):
        raise self.err

@with_seq_params
def test_ramp_eval_error1(max_bt):
    s = new_seq(max_bt)

    f1 = ErrorFunction(RuntimeError("AAAA-BBBB.CCCC"))
    s.add_step(0.1).pulse('artiq/analog', 0.1)
    s.wait(0.2)
    def asd7f8ashdfasd():
        s.add_step(0.1).set('artiq/analog', f1)
    asd7f8ashdfasd()

    assert str(s) == f"""Seq - T[3]
 T[0]: 0 ps
 T[1]: T[0] + 100 ms
 T[2]: T[1] + 200 ms
 T[3]: T[2] + 100 ms
  TimeStep(0.1)@T[0]
    artiq/analog: Pulse(0.1)
  TimeStep(0.1)@T[2]
    artiq/analog: Set({f1})
"""

    with pytest.raises(RuntimeError, match="AAAA-BBBB.CCCC") as exc:
        test_utils.seq_finalize(s)
    check_bt(exc, max_bt, 'asd7f8ashdfasd')
