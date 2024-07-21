#

import dummy_pulse_compiler
dummy_pulse_compiler.inject()

from brassboard_seq.action import RampFunction
from brassboard_seq.config import Config
from brassboard_seq import backend, rfsoc_backend, rtval, seq
import brassboard_seq_test_utils as test_utils
import brassboard_seq_rfsoc_backend_utils as rfsoc_utils

import pytest

global_conf = Config()
global_conf.add_supported_prefix('artiq')
global_conf.add_supported_prefix('rfsoc')

def new_seq_compiler(*args):
    s = seq.Seq(global_conf, *args)
    comp = backend.SeqCompiler(s)
    return s, comp

global_gen = rfsoc_backend.PulseCompilerGenerator()

def add_rfsoc_backend(comp):
    rb = rfsoc_backend.RFSOCBackend(global_gen)
    comp.add_backend('rfsoc', rb)
    comp.add_backend('artiq', backend.Backend()) # Dummy backend
    return rb

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

class StaticFunction(RampFunction):
    def __init__(self):
        RampFunction.__init__(self)

    def eval(self, t, length, oldval):
        return t / 2 + oldval - length

def test_generator():
    rfsoc_backend.PulseCompilerGenerator()
    rfsoc_backend.PulseCompilerGenerator()

def get_channel_info(rb, s):
    channels = rfsoc_utils.get_channel_info(rb)
    compiled_info = rfsoc_utils.get_compiled_info(rb)
    chn_ids = set(tone_chn.chn for tone_chn in channels.channels)
    assert len(chn_ids) == len(channels.channels)
    chn_params = set()
    for seq_chn, (chn_idx, param) in channels.chn_map.items():
        assert param in ('amp', 'freq', 'phase', 'ff')
        assert 0 <= chn_idx
        assert chn_idx < len(channels.channels)
        assert (chn_idx, param) not in chn_params
        chn_params.add((chn_idx, param))

    all_actions = {}
    for (chn, actions) in enumerate(test_utils.seq_get_all_actions(s)):
        for action in actions:
            if test_utils.action_get_cond(action) is False:
                continue
            all_actions[test_utils.action_get_aid(action)] = (chn, action,
                                                                  [False, False])
    bool_values_used = [False for _ in range(len(compiled_info.bool_values))]
    float_values_used = [False for _ in range(len(compiled_info.float_values))]
    relocations_used = [False for _ in range(len(compiled_info.relocations))]

    for chn in channels.channels:
        for (param, param_actions) in enumerate(chn.actions):
            param = ['freq', 'amp', 'phase', 'ff'][param]
            for rfsoc_action in param_actions:
                chn, action, seen = all_actions[rfsoc_action.aid]
                action_info = test_utils.action_get_compile_info(action)
                if rfsoc_action.reloc_id >= 0:
                    reloc = compiled_info.relocations[rfsoc_action.reloc_id]
                    relocations_used[rfsoc_action.reloc_id] = True
                    assert (reloc.cond_idx >= 0 or reloc.val_idx >= 0 or
                                reloc.time_idx >= 0)
                    cond_idx = reloc.cond_idx
                    val_idx = reloc.val_idx
                    time_idx = reloc.time_idx
                else:
                    cond_idx = -1
                    val_idx = -1
                    time_idx = -1

                cond = test_utils.action_get_cond(action)
                if cond_idx >= 0:
                    bool_values_used[cond_idx] = True
                    assert isinstance(cond, rtval.RuntimeValue)
                    assert compiled_info.bool_values[cond_idx] is cond
                else:
                    assert cond

                event_time = test_utils.seq_get_event_time(s, rfsoc_action.tid)
                static_time = test_utils.event_time_get_static(event_time)
                if time_idx >= 0:
                    assert static_time == -1
                else:
                    assert static_time == rfsoc_action.seq_time

                action_value = test_utils.action_get_value(action)
                isramp = isinstance(action_value, RampFunction)
                if rfsoc_action.tid == action_info['tid']:
                    assert not seen[0]
                    seen[0] = True
                    value = action_value
                    assert isramp == rfsoc_action.isramp
                else:
                    assert rfsoc_action.tid == action_info['end_tid']
                    isramp = isinstance(test_utils.action_get_value(action),
                                        RampFunction)
                    assert test_utils.action_get_is_pulse(action) or isramp
                    assert not seen[1]
                    seen[1] = True
                    value = action_info['end_val']
                # Check value
                if val_idx >= 0:
                    if rfsoc_action.isramp:
                        length = action_info['length']
                        assert isinstance(length, rtval.RuntimeValue)
                        float_values_used[val_idx] = True
                        assert compiled_info.float_values[val_idx] is length
                    else:
                        assert isinstance(value, rtval.RuntimeValue)
                        if param == 'ff':
                            bool_values_used[val_idx] = True
                            assert compiled_info.bool_values[val_idx] is value
                        else:
                            float_values_used[val_idx] = True
                            assert compiled_info.float_values[val_idx] is value
                elif rfsoc_action.isramp:
                    length = action_info['length']
                    assert not isinstance(length, rtval.RuntimeValue)
                    assert rfsoc_action.float_value == length
                else:
                    assert not isinstance(value, rtval.RuntimeValue)
                    if param == 'ff':
                        assert rfsoc_action.bool_value == value
                    else:
                        assert rfsoc_action.float_value == value

    for chn, action, seen in all_actions.values():
        assert seen[0]
        action_value = test_utils.action_get_value(action)
        isramp = isinstance(action_value, RampFunction)
        if test_utils.action_get_is_pulse(action) or isramp:
            assert seen[1]
        else:
            assert not seen[1]
    assert all(bool_values_used)
    assert all(float_values_used)
    assert all(relocations_used)

    return channels

with_rfsoc_params = with_params((0,), (5,), (500,))

@with_rfsoc_params
def test_channels(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    assert s.get_channel_id('artiq/ttl0') == 0
    assert s.get_channel_id('rfsoc/dds1/0/amp') == 1
    assert s.get_channel_id('rfsoc/dds1/1/freq') == 2
    assert s.get_channel_id('artiq/ttl2') == 3
    assert s.get_channel_id('rfsoc/dds2/0/phase') == 4
    assert s.get_channel_id('rfsoc/dds0/0/ff') == 5
    assert s.get_channel_id('rfsoc/dds1/1/amp') == 6
    assert s.get_channel_id('rfsoc/dds1/1/ff') == 7
    assert s.get_channel_id('rfsoc/dds0/0/freq') == 8
    comp.finalize()
    channels = get_channel_info(rb, s)
    chn_ids = [tone_chn.chn for tone_chn in channels.channels]
    assert chn_ids == [2, 3, 4, 0]
    assert channels.chn_map == {1: (0, 'amp'), 2: (1, 'freq'),
                                4: (2, 'phase'), 5: (3, 'ff'),
                                6: (1, 'amp'), 7: (1, 'ff'),
                                8: (3, 'freq')}

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    assert s.get_channel_id('rfsoc/dds0/0') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds0/0'):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    assert s.get_channel_id('rfsoc/dds0/0/freq/a') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds0/0/freq/a'):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    assert s.get_channel_id('rfsoc/ch0/0/freq') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/ch0/0/freq'):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    assert s.get_channel_id('rfsoc/dds50/0/freq') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds50/0/freq'):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    assert s.get_channel_id('rfsoc/dds10/2/freq') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds10/2/freq'):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    assert s.get_channel_id('rfsoc/dds10/0/param') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds10/0/param'):
        comp.finalize()

@with_rfsoc_params
def test_output1(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.add_step(0.01) \
      .pulse('rfsoc/dds0/1/amp', 0.2) \
      .set('rfsoc/dds0/1/freq', 100e6) \
      .set('rfsoc/dds0/1/phase', 0.1) \
      .set('rfsoc/dds0/1/ff', True)
    comp.finalize()
    channels = get_channel_info(rb, s)
    assert len(channels.channels) == 1
    assert channels.channels[0].chn == 1
    assert len(channels.channels[0].actions[0]) == 1 # freq
    action = channels.channels[0].actions[0][0]
    assert action.cond
    assert not action.isramp
    assert not action.sync
    assert action.reloc_id == -1
    assert action.aid == 1
    assert action.tid == 0
    assert not action.is_end
    assert action.seq_time == 0
    assert action.float_value == 100e6
    assert len(channels.channels[0].actions[1]) == 1 # phase
    action = channels.channels[0].actions[1][0]
    assert action.cond
    assert not action.isramp
    assert not action.sync
    assert action.reloc_id == -1
    assert action.aid == 2
    assert action.tid == 0
    assert not action.is_end
    assert action.seq_time == 0
    assert action.float_value == 0.1
    assert len(channels.channels[0].actions[2]) == 2 # amp
    action = channels.channels[0].actions[2][0]
    assert action.cond
    assert not action.isramp
    assert not action.sync
    assert action.reloc_id == -1
    assert action.aid == 0
    assert action.tid == 0
    assert not action.is_end
    assert action.seq_time == 0
    assert action.float_value == 0.2
    action = channels.channels[0].actions[2][1]
    assert action.cond
    assert not action.isramp
    assert not action.sync
    assert action.reloc_id == -1
    assert action.aid == 0
    assert action.tid == 1
    assert action.is_end
    assert action.seq_time == 10_000_000_000
    assert action.float_value == 0.0
    assert len(channels.channels[0].actions[3]) == 1 # ff
    action = channels.channels[0].actions[3][0]
    assert action.cond
    assert not action.isramp
    assert not action.sync
    assert action.reloc_id == -1
    assert action.aid == 3
    assert action.tid == 0
    assert not action.is_end
    assert action.seq_time == 0
    assert action.bool_value

@with_rfsoc_params
def test_output2(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.conditional(rtval.new_extern(lambda: True)) \
      .add_step(rtval.new_extern(lambda: 0.01)) \
      .pulse('rfsoc/dds0/1/amp', rtval.new_extern(lambda: 0.2)) \
      .set('rfsoc/dds0/1/freq', rtval.new_extern(lambda: 100e6), sync=True) \
      .set('rfsoc/dds0/1/phase', rtval.new_extern(lambda: 0.1)) \
      .set('rfsoc/dds0/1/ff', rtval.new_extern(lambda: True), sync=False)
    comp.finalize()
    channels = get_channel_info(rb, s)
    assert len(channels.channels) == 1
    assert channels.channels[0].chn == 1
    assert len(channels.channels[0].actions[0]) == 1 # freq
    action = channels.channels[0].actions[0][0]
    assert not action.isramp
    assert action.sync
    assert action.reloc_id != -1
    assert action.aid == 1
    assert action.tid == 0
    assert not action.is_end
    assert action.seq_time == 0
    assert len(channels.channels[0].actions[1]) == 1 # phase
    action = channels.channels[0].actions[1][0]
    assert not action.isramp
    assert not action.sync
    assert action.reloc_id != -1
    assert action.aid == 2
    assert action.tid == 0
    assert not action.is_end
    assert action.seq_time == 0
    assert len(channels.channels[0].actions[2]) == 2 # amp
    action = channels.channels[0].actions[2][0]
    assert not action.isramp
    assert not action.sync
    assert action.reloc_id != -1
    assert action.aid == 0
    assert action.tid == 0
    assert not action.is_end
    assert action.seq_time == 0
    action = channels.channels[0].actions[2][1]
    assert not action.isramp
    assert not action.sync
    assert action.reloc_id != -1
    assert action.aid == 0
    assert action.tid == 1
    assert action.is_end
    assert len(channels.channels[0].actions[3]) == 1 # ff
    action = channels.channels[0].actions[3][0]
    assert not action.isramp
    assert not action.sync
    assert action.reloc_id != -1
    assert action.aid == 3
    assert action.tid == 0
    assert not action.is_end
    assert action.seq_time == 0

@with_rfsoc_params
def test_output3(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.conditional(False) \
      .add_step(rtval.new_extern(lambda: 0.01)) \
      .pulse('rfsoc/dds0/1/amp', rtval.new_extern(lambda: 0.2)) \
      .set('rfsoc/dds0/1/freq', rtval.new_extern(lambda: 100e6), sync=True) \
      .set('rfsoc/dds0/1/phase', rtval.new_extern(lambda: 0.1)) \
      .set('rfsoc/dds0/1/ff', rtval.new_extern(lambda: True), sync=False)
    comp.finalize()
    channels = get_channel_info(rb, s)
    assert len(channels.channels) == 1
    assert channels.channels[0].chn == 1
    assert len(channels.channels[0].actions[0]) == 0 # freq
    assert len(channels.channels[0].actions[1]) == 0 # phase
    assert len(channels.channels[0].actions[2]) == 0 # amp
    assert len(channels.channels[0].actions[3]) == 0 # ff

@with_rfsoc_params
def test_ramp_output1(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    ramp1 = StaticFunction()
    ramp2 = StaticFunction()
    s.add_step(0.01) \
      .pulse('rfsoc/dds0/1/amp', ramp1) \
      .set('rfsoc/dds0/1/phase', ramp2)
    comp.finalize()
    channels = get_channel_info(rb, s)
    assert len(channels.channels) == 1
    assert channels.channels[0].chn == 1
    assert len(channels.channels[0].actions[0]) == 0
    assert len(channels.channels[0].actions[1]) == 2 # phase
    action = channels.channels[0].actions[1][0]
    assert action.cond
    assert action.isramp
    assert not action.sync
    assert action.reloc_id == -1
    assert action.aid == 1
    assert action.tid == 0
    assert not action.is_end
    assert action.seq_time == 0
    assert action.float_value == 0.01
    assert action.ramp is ramp2
    action = channels.channels[0].actions[1][1]
    assert action.cond
    assert not action.isramp
    assert not action.sync
    assert action.reloc_id == -1
    assert action.aid == 1
    assert action.tid == 1
    assert action.is_end
    assert action.seq_time == 10_000_000_000
    assert action.float_value == -0.005
    assert len(channels.channels[0].actions[2]) == 2 # amp
    action = channels.channels[0].actions[2][0]
    assert action.cond
    assert action.isramp
    assert not action.sync
    assert action.reloc_id == -1
    assert action.aid == 0
    assert action.tid == 0
    assert not action.is_end
    assert action.seq_time == 0
    assert action.float_value == 0.01
    assert action.ramp is ramp1
    action = channels.channels[0].actions[2][1]
    assert action.cond
    assert not action.isramp
    assert not action.sync
    assert action.reloc_id == -1
    assert action.aid == 0
    assert action.tid == 1
    assert action.is_end
    assert action.seq_time == 10_000_000_000
    assert action.float_value == 0.0
    assert len(channels.channels[0].actions[3]) == 0 # ff

@with_rfsoc_params
def test_ramp_output2(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    ramp1 = StaticFunction()
    ramp2 = StaticFunction()
    s.add_step(rtval.new_extern(lambda: 0.01)) \
      .pulse('rfsoc/dds0/1/amp', ramp1) \
      .set('rfsoc/dds0/1/phase', ramp2)
    comp.finalize()
    channels = get_channel_info(rb, s)
    assert len(channels.channels) == 1
    assert channels.channels[0].chn == 1
    assert len(channels.channels[0].actions[0]) == 0 # freq
    assert len(channels.channels[0].actions[1]) == 2 # phase
    action = channels.channels[0].actions[1][0]
    assert action.cond
    assert action.isramp
    assert not action.sync
    assert action.reloc_id != -1
    assert action.aid == 1
    assert action.tid == 0
    assert not action.is_end
    assert action.seq_time == 0
    assert action.ramp is ramp2
    action = channels.channels[0].actions[1][1]
    assert action.cond
    assert not action.isramp
    assert not action.sync
    assert action.reloc_id != -1
    assert action.aid == 1
    assert action.tid == 1
    assert action.is_end
    assert len(channels.channels[0].actions[2]) == 2 # amp
    action = channels.channels[0].actions[2][0]
    assert action.cond
    assert action.isramp
    assert not action.sync
    assert action.reloc_id != -1
    assert action.aid == 0
    assert action.tid == 0
    assert not action.is_end
    assert action.seq_time == 0
    assert action.ramp is ramp1
    action = channels.channels[0].actions[2][1]
    assert action.cond
    assert not action.isramp
    assert not action.sync
    assert action.reloc_id != -1
    assert action.aid == 0
    assert action.tid == 1
    assert action.is_end
    assert len(channels.channels[0].actions[3]) == 0 # ff

@with_rfsoc_params
def test_arg_error(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    st = s.add_step(0.01)
    def jasd89jfkalsdfasd():
        st.pulse('rfsoc/dds0/1/amp', 0.2, sth=0.1)
    jasd89jfkalsdfasd()
    with pytest.raises(ValueError,
                       match="Invalid output keyword argument {'sth': 0.1}") as exc:
        comp.finalize()
    check_bt(exc, max_bt, 'jasd89jfkalsdfasd')

@with_rfsoc_params
def test_ff_ramp(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    st = s.add_step(0.01)
    def ja98fas923ncf():
        st.pulse('rfsoc/dds0/1/ff', StaticFunction())
    ja98fas923ncf()
    with pytest.raises(ValueError,
                       match="Feed forward control cannot be ramped") as exc:
        comp.finalize()
    check_bt(exc, max_bt, 'ja98fas923ncf')

@with_rfsoc_params
def test_val_error(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    # This causes a error to be thrown when converting to boolean
    def j893ajjaks988394():
        s.set('rfsoc/dds0/0/ff', np.array([1, 2]))
    j893ajjaks988394()
    with pytest.raises(ValueError) as exc:
        comp.finalize()
    check_bt(exc, max_bt, 'j893ajjaks988394')

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    # This causes a error to be thrown when converting to float
    def a88f2398fasd():
        s.set('rfsoc/dds3/0/freq', [1, 2])
    a88f2398fasd()
    with pytest.raises(TypeError) as exc:
        comp.finalize()
    check_bt(exc, max_bt, 'a88f2398fasd')
