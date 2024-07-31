#

import dummy_pulse_compiler
dummy_pulse_compiler.inject()

from brassboard_seq.action import Blackman, LinearRamp, RampFunction, SeqCubicSpline
from brassboard_seq.config import Config
from brassboard_seq import backend, rfsoc_backend, rtval, seq
import brassboard_seq_test_utils as test_utils
import brassboard_seq_rfsoc_backend_utils as rfsoc_utils

import pytest
import typing
import numpy as np

class Spline(typing.NamedTuple):
    order0: float
    order1: float = 0.0
    order2: float = 0.0
    order3: float = 0.0

class Tone(typing.NamedTuple):
    cycles: int
    freq: tuple
    amp: tuple
    phase: tuple
    sync: bool
    feedback: bool

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

def get_output():
    output = global_gen.output
    from qiskit.pulse import ControlChannel, DriveChannel

    total_len = None

    stripped_output = {}

    for (key, tonedatas) in output.items():
        if isinstance(key, ControlChannel):
            assert key.index in range(2)
            chn = key.index
        else:
            assert isinstance(key, DriveChannel)
            assert key.index in range(62)
            chn = key.index + 2
        tone_len = 0
        is_first = True
        stripped_tonedatas = []
        stripped_output[chn] = stripped_tonedatas
        for tonedata in tonedatas:
            assert tonedata.tone in range(2)
            assert tonedata.channel in range(32)
            assert (tonedata.channel << 1) | tonedata.tone == chn
            assert tonedata.duration_cycles >= 4
            tone_len += tonedata.duration_cycles
            assert tonedata.frame_rotation_rad == (0, 0, 0, 0)

            assert tonedata.wait_trigger == is_first
            is_first = False
            assert not tonedata.output_enable
            assert not tonedata.bypass_lookup_tables

            stripped_tonedatas.append(Tone(tonedata.duration_cycles,
                                           Spline(*tonedata.frequency_hz),
                                           Spline(*tonedata.amplitude),
                                           Spline(*tonedata.phase_rad),
                                           tonedata.sync, tonedata.feedback_enable))
        if total_len is None:
            total_len = tone_len
        else:
            assert total_len == tone_len
        assert total_len >= 8

    return stripped_output

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

class LinearRampNoSeg(RampFunction):
    def __init__(self, start, end):
        super().__init__(start=start, end=end)

    def eval(self, t, length, oldval):
        t = t / length
        return self.start * (1 - t) + self.end * t

class RampUpAndDown(RampFunction):
    def __init__(self, pos, val):
        super().__init__(pos=pos, val=val)

    def eval(self, t, length, oldval):
        t = t / length
        len1 = self.pos
        len2 = 1 - self.pos
        t1 = t / len1
        t2 = (t - self.pos) / len2
        v1 = oldval * (1 - t1) + self.val * t1
        v2 = self.val * (1 - t2) + oldval * t2
        return rtval.ifelse(t > self.pos, v2, v1)

    def spline_segments(self, length, oldval):
        return (length * self.pos,)

class ErrorSegment(RampFunction):
    def __init__(self):
        super().__init__()

    def eval(self, t, length, oldval):
        return t

    def spline_segments(self, length, oldval):
        raise ValueError("JJJLLL---DFFDAFD")

class ErrorEval(RampFunction):
    def __init__(self):
        super().__init__()

    def eval(self, t, length, oldval):
        raise ValueError("XXXJJFSJkdfFDSDF")

class DivLengthFunction(RampFunction):
    def __init__(self):
        super().__init__()

    def eval(self, t, length, oldval):
        return t / length

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
def test_generator_error(max_bt):
    s, comp = new_seq_compiler(max_bt)
    gen = rfsoc_utils.ErrorGenerator()
    rb = rfsoc_backend.RFSOCBackend(gen)
    comp.add_backend('rfsoc', rb)
    assert s.get_channel_id('rfsoc/dds1/0/amp') == 0
    comp.finalize()

    with pytest.raises(RuntimeError, match="AAABBBCCCC Error Generator DDDEEEFFF"):
        comp.runtime_finalize(1)

@with_rfsoc_params
def test_channels(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    assert not rb.has_output
    comp.finalize()
    assert not rb.has_output

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
    assert not rb.has_output
    comp.finalize()
    assert rb.has_output
    channels = get_channel_info(rb, s)
    chn_ids = [tone_chn.chn for tone_chn in channels.channels]
    assert chn_ids == [2, 3, 4, 0]
    assert channels.chn_map == {1: (0, 'amp'), 2: (1, 'freq'),
                                4: (2, 'phase'), 5: (3, 'ff'),
                                6: (1, 'amp'), 7: (1, 'ff'),
                                8: (3, 'freq')}
    comp.runtime_finalize(1)
    assert get_output() == {
        0: [Tone(8, Spline(0.0), Spline(0.0), Spline(0.0), False, False)],
        2: [Tone(8, Spline(0.0), Spline(0.0), Spline(0.0), False, False)],
        3: [Tone(8, Spline(0.0), Spline(0.0), Spline(0.0), False, False)],
        4: [Tone(8, Spline(0.0), Spline(0.0), Spline(0.0), False, False)],
    }

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

    comp.runtime_finalize(1)
    assert get_output() == {
        1: [Tone(4096000, Spline(100e6), Spline(0.2),
                 Spline(0.2 * np.pi), False, True),
            Tone(8, Spline(100e6), Spline(0.0), Spline(0.2 * np.pi), False, True)],
    }

@with_rfsoc_params
def test_output2(max_bt):
    b1 = True
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.conditional(rtval.new_extern(lambda: b1)) \
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

    comp.runtime_finalize(1)
    assert get_output() == {
        1: [Tone(4096000, Spline(100e6), Spline(0.2),
                 Spline(0.2 * np.pi), True, True),
            Tone(8, Spline(100e6), Spline(0.0), Spline(0.2 * np.pi), False, True)],
    }

    b1 = False
    comp.runtime_finalize(2)
    assert get_output() == {
        1: [Tone(8, Spline(0.0), Spline(0.0), Spline(0.0), False, False)],
    }

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

    comp.runtime_finalize(1)
    assert get_output() == {
        1: [Tone(8, Spline(0.0), Spline(0.0), Spline(0.0), False, False)],
    }

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

    comp.runtime_finalize(1)
    assert get_output() == {
        1: [Tone(2048000, Spline(0.0), pytest.approx(Spline(-0.01, 0.0025)),
                 pytest.approx(Spline(-0.02 * np.pi, 0.005 * np.pi)), False, False),
            Tone(2048000, Spline(0.0), pytest.approx(Spline(-0.0075, 0.0025)),
                 pytest.approx(Spline(-0.015 * np.pi, 0.005 * np.pi)), False, False),
            Tone(8, Spline(0.0), Spline(0.0), pytest.approx(Spline(-0.01 * np.pi)),
                 False, False)],
    }

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

    comp.runtime_finalize(1)
    assert get_output() == {
        1: [Tone(2048000, Spline(0.0), pytest.approx(Spline(-0.01, 0.0025)),
                 pytest.approx(Spline(-0.02 * np.pi, 0.005 * np.pi)), False, False),
            Tone(2048000, Spline(0.0), pytest.approx(Spline(-0.0075, 0.0025)),
                 pytest.approx(Spline(-0.015 * np.pi, 0.005 * np.pi)), False, False),
            Tone(8, Spline(0.0), Spline(0.0), pytest.approx(Spline(-0.01 * np.pi)),
                 False, False)],
    }

@with_rfsoc_params
def test_ramp_output3(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    ramp1 = LinearRampNoSeg(0.1, 0.2)
    ramp2 = LinearRamp(0.1, 0.2)
    ramp3 = SeqCubicSpline(0.2, 0.1, 0.9, 0.3)
    s.add_step(5e-3) \
      .pulse('rfsoc/dds0/1/amp', ramp1) \
      .set('rfsoc/dds2/1/phase', ramp2) \
      .set('rfsoc/dds4/1/amp', ramp3) \
      .set('rfsoc/dds2/0/amp', 0.1)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        1: [Tone(1024000, Spline(0.0), pytest.approx(Spline(0.1, 0.05, 0.0, 0.0)),
                 Spline(0.0), False, False),
            Tone(1024000, Spline(0.0), pytest.approx(Spline(0.15, 0.05, 0.0, 0.0)),
                 Spline(0.0), False, False),
            Tone(8, Spline(0.0), Spline(0.0), Spline(0.0), False, False)],
        4: [Tone(2048008, Spline(0.0), Spline(0.1), Spline(0.0), False, False)],
        5: [Tone(2048000, Spline(0.0), Spline(0.0),
                 pytest.approx(Spline(0.2 * np.pi, 0.2 * np.pi)),
                 False, False),
            Tone(8, Spline(0.0), Spline(0.0), Spline(0.4 * np.pi), False, False)],
        9: [Tone(2048000, Spline(0.0), Spline(0.2, 0.1, 0.9, 0.3),
                 Spline(0.0), False, False),
            Tone(8, Spline(0.0), Spline(1.5), Spline(0.0), False, False)],
    }

@with_rfsoc_params
def test_ramp_output4(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    ramp1 = LinearRampNoSeg(0.1, 0.2)
    ramp2 = SeqCubicSpline(0.2, 0.1, 0.9, 0.3)
    s.add_step(5e-3) \
      .pulse('rfsoc/dds0/1/phase', ramp1) \
      .set('rfsoc/dds2/1/phase', ramp2) \
      .set('rfsoc/dds2/0/amp', 0.1) \
      .set('rfsoc/dds2/0/phase', -1 / 2 / np.pi)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        1: [Tone(1024000, Spline(0.0), Spline(0.0),
                 pytest.approx(Spline(0.2 * np.pi, 0.1 * np.pi)), False, False),
            Tone(1024000, Spline(0.0), Spline(0.0),
                 pytest.approx(Spline(0.3 * np.pi, 0.1 * np.pi)), False, False),
            Tone(8, Spline(0.0), Spline(0.0), Spline(0.0), False, False)],
        4: [Tone(2048008, Spline(0.0), Spline(0.1), Spline(-1.0), False, False)],
        5: [Tone(2048000, Spline(0.0), Spline(0.0),
                 pytest.approx(Spline(0.4 * np.pi, 0.2 * np.pi, 1.8 * np.pi, 0.6 * np.pi)),
                 False, False),
            Tone(8, Spline(0.0), Spline(0.0), Spline(3 * np.pi), False, False)],
    }

@with_rfsoc_params
def test_ramp_output5(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.set('rfsoc/dds0/1/amp', 0.2)
    s.set('rfsoc/dds2/0/freq', 80e6)
    s.set('rfsoc/dds0/1/phase', 0.9)
    s.add_step(5e-3) \
      .set('rfsoc/dds0/1/amp', RampUpAndDown(0.25, 0.5)) \
      .set('rfsoc/dds2/0/freq', RampUpAndDown(0.5, 100e6)) \
      .set('rfsoc/dds0/1/phase', RampUpAndDown(0.75, 0.3)) \
      .set('rfsoc/dds0/0/amp', 0.2) \
      .set('rfsoc/dds0/0/phase', -0.5 / 2 / np.pi)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        0: [Tone(2048008, Spline(0.0), Spline(0.2), Spline(-0.5), False, False)],
        1: [Tone(512000, Spline(0.0), pytest.approx(Spline(0.2, 0.3)),
                 pytest.approx(Spline(1.8 * np.pi, -0.4 * np.pi)), False, False),
            Tone(1024000, Spline(0.0), pytest.approx(Spline(0.5, -0.2)),
                 pytest.approx(Spline(1.4 * np.pi, -0.8 * np.pi)), False, False),
            Tone(512000, Spline(0.0), pytest.approx(Spline(0.3, -0.1)),
                 pytest.approx(Spline(0.6 * np.pi, 1.2 * np.pi)), False, False),
            Tone(8, Spline(0.0), Spline(0.2), Spline(1.8 * np.pi), False, False)],
        4: [Tone(1024000, pytest.approx(Spline(80e6, 20e6), abs=1e-3),
                 Spline(0.0), Spline(0.0), False, False),
            Tone(1024000, pytest.approx(Spline(100e6, -20e6), abs=1e-3),
                 Spline(0.0), Spline(0.0), False, False),
            Tone(8, Spline(80e6), Spline(0.0), Spline(0.0), False, False)],
    }

@with_rfsoc_params
def test_ramp_output6(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    @s.add_background
    def amp_step(s):
        s.add_step(4e-3) \
         .set('rfsoc/dds0/0/amp', SeqCubicSpline(0.2, -0.1, 0.9, -0.3))
        s.add_step(4e-3) \
         .set('rfsoc/dds0/0/amp', SeqCubicSpline(0.5, -0.1, 0.5, -0.2))
    s.wait(1e-3)
    s.set('rfsoc/dds2/0/amp', 0.1)
    @s.add_background
    def freq_step(s):
        s.add_step(4e-3) \
         .set('rfsoc/dds0/0/freq', SeqCubicSpline(100e6, 10e6, -20e6, 20e6))
        s.add_step(4e-3) \
         .set('rfsoc/dds0/0/freq', SeqCubicSpline(10e6, 100e6, -30e6, -20e6))
    s.wait(1e-3)
    @s.add_background
    def phase_step(s):
        s.add_step(4e-3) \
         .set('rfsoc/dds0/0/phase', SeqCubicSpline(0.4, -0.2, 0.1, -0.2))
        s.add_step(4e-3) \
         .set('rfsoc/dds0/0/phase', SeqCubicSpline(0.2, 0.7, 0.1, -0.9))
    s.wait(1e-3)
    @s.add_background
    def phase_step(s):
        s.add_step(4e-3) \
         .set('rfsoc/dds0/0/ff', True)
        s.add_step(4e-3) \
         .set('rfsoc/dds0/0/ff', False)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        0: [
            # amp 0.0
            Tone(409600, Spline(0.0),
                 pytest.approx(Spline(0.2, -0.025, 0.05625, -0.0046875)),
                 Spline(0.0), False, False),
            # amp 0.1
            # freq 0.0
            Tone(409600,
                 pytest.approx(Spline(100e6, 2.5e6, -1.25e6, 0.3125e6)),
                 pytest.approx(Spline(0.2265625, 0.0734375, 0.0421875, -0.0046875)),
                 Spline(0.0), False, False),
            # amp 0.2
            # freq 0.1
            # phase 0.0
            Tone(409600,
                 pytest.approx(Spline(101.5625e6, 0.9375e6, -0.3125e6, 0.3125e6)),
                 pytest.approx(Spline(0.3375, 0.14375, 0.028125, -0.0046875)),
                 pytest.approx(Spline(0.8 * np.pi, -0.1 * np.pi,
                                      0.0125 * np.pi, -0.00625 * np.pi)),
                 False, False),
            # amp 0.3
            # freq 0.2
            # phase 0.1
            Tone(409600,
                 pytest.approx(Spline(102.5e6, 1.25e6, 0.625e6, 0.3125e6)),
                 pytest.approx(Spline(0.5046875, 0.1859375, 0.0140625, -0.0046875)),
                 pytest.approx(Spline(0.70625 * np.pi, -0.09375 * np.pi,
                                      -0.00625 * np.pi, -0.00625 * np.pi)),
                 False, True),
            # amp 1.0
            # freq 0.3
            # phase 0.2
            Tone(409600,
                 pytest.approx(Spline(104.6875e6, 3.4375e6, 1.5625e6, 0.3125e6)),
                 pytest.approx(Spline(0.5, -0.025, 0.03125, -0.003125)),
                 pytest.approx(Spline(0.6 * np.pi, -0.125 * np.pi,
                                      -0.025 * np.pi, -0.00625 * np.pi)),
                 False, True),
            # amp 1.1
            # freq 1.0
            # phase 0.3
            Tone(409600,
                 pytest.approx(Spline(10e6, 25e6, -1.875e6, -0.3125e6)),
                 pytest.approx(Spline(0.503125, 0.028125, 0.021875, -0.003125)),
                 pytest.approx(Spline(0.44375 * np.pi, -0.19375 * np.pi,
                                      -0.04375 * np.pi, -0.00625 * np.pi)),
                 False, True),
            # amp 1.2
            # freq 1.1
            # phase 1.0
            Tone(409600,
                 pytest.approx(Spline(32.8125e6, 20.3125e6, -2.8125e6, -0.3125e6)),
                 pytest.approx(Spline(0.55, 0.0625, 0.0125, -0.003125)),
                 pytest.approx(Spline(0.4 * np.pi, 0.35 * np.pi,
                                      0.0125 * np.pi, -0.028125 * np.pi)),
                 False, True),
            # amp 1.3
            # freq 1.2
            # phase 1.1
            Tone(409600,
                 pytest.approx(Spline(50e6, 13.75e6, -3.75e6, -0.3125e6)),
                 pytest.approx(Spline(0.621875, 0.078125, 0.003125, -0.003125)),
                 pytest.approx(Spline(0.734375 * np.pi, 0.290625 * np.pi,
                                      -0.071875 * np.pi, -0.028125 * np.pi)),
                 False, False),
            # freq 1.3
            # phase 1.2
            Tone(409600,
                 pytest.approx(Spline(59.6875e6, 5.3125e6, -4.6875e6, -0.3125e6)),
                 Spline(0.7),
                 pytest.approx(Spline(0.925 * np.pi, 0.0625 * np.pi,
                                      -0.15625 * np.pi, -0.028125 * np.pi)),
                 False, False),
            # phase 1.3
            Tone(409600, Spline(60e6), Spline(0.7),
                 pytest.approx(Spline(0.803125 * np.pi, -0.334375 * np.pi,
                                      -0.240625 * np.pi, -0.028125 * np.pi)),
                 False, False),
            Tone(409608, Spline(60e6), Spline(0.7),
                 pytest.approx(Spline(0.2 * np.pi)), False, False)],
        4: [Tone(409600, Spline(0.0), Spline(0.0), Spline(0.0), False, False),
            Tone(4096008, Spline(0.0), Spline(0.1), Spline(0.0), False, False)],
    }

@with_rfsoc_params
def test_ramp_output7(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.add_step(1) \
      .set('rfsoc/dds0/0/amp', Blackman(1.0)) \
      .set('rfsoc/dds0/1/amp', 1.0)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        0: [
            Tone(25600000, Spline(0.0),
                 pytest.approx(Spline(0.0, 0.00015745877326822644,
                                      0.01300285490625988, 0.001468462559752276)),
                 Spline(0.0), False, False),
            Tone(25600000, Spline(0.0),
                 pytest.approx(Spline(0.014628776239280439, 0.03074418247908131,
                                      0.01795439872470489, 0.003119251963659586)),
                 Spline(0.0), False, False),
            Tone(25600000, Spline(0.0),
                 pytest.approx(Spline(0.06644660940672625, 0.07585724657247933,
                                      0.028056006241789055, 0.0017298791015365467)),
                 Spline(0.0), False, False),
            Tone(25600000, Spline(0.0),
                 pytest.approx(Spline(0.17208974132253135, 0.13669965899911896,
                                      0.033698736281407626, -0.002488136603058244)),
                 Spline(0.0), False, False),
            Tone(25600000, Spline(0.0),
                 pytest.approx(Spline(0.34, 0.19610073763144964,
                                      0.026062367723541868, -0.007389931667371563)),
                 Spline(0.0), False, False),
            Tone(25600000, Spline(0.0),
                 pytest.approx(Spline(0.5547731736876211, 0.22576257627864393,
                                      0.003121741649712906, -0.010104101022703649)),
                 Spline(0.0), False, False),
            Tone(25600000, Spline(0.0),
                 pytest.approx(Spline(0.7735533905932739, 0.20184724575694735,
                                      -0.028177599439071166, -0.008714728160579277)),
                 Spline(0.0), False, False),
            Tone(25600000, Spline(0.0),
                 pytest.approx(Spline(0.9385083087505672, 0.11992457534234635,
                                      -0.05499955158835057, -0.0034333325045636798)),
                 Spline(0.0), False, False),
            Tone(25600000, Spline(0.0),
                 pytest.approx(Spline(1.0, 0.00037452534804338633,
                                      -0.06529954910203895, 0.003433332504564568)),
                 Spline(0.0), False, False),
            Tone(25600000, Spline(0.0),
                 pytest.approx(Spline(0.9385083087505672, -0.11934786239705719,
                                      -0.05432178392081566, 0.008714728160582386)),
                 Spline(0.0), False, False),
            Tone(25600000, Spline(0.0),
                 pytest.approx(Spline(0.7735533905932739, -0.2016937565099568,
                                      -0.027190561418399817, 0.010104101022704093)),
                 Spline(0.0), False, False),
            Tone(25600000, Spline(0.0),
                 pytest.approx(Spline(0.5547731736876214, -0.22605567807641802,
                                      0.0038925727214200734, 0.007389931667376892)),
                 Spline(0.0), False, False),
            Tone(25600000, Spline(0.0),
                 pytest.approx(Spline(0.34, -0.19663272175275923,
                                      0.026234326472228675, 0.002488136603061575)),
                 Spline(0.0), False, False),
            Tone(25600000, Spline(0.0),
                 pytest.approx(Spline(0.17208974132253121, -0.13715889636066686,
                                      0.03324564354639803, -0.0017298791015359916)),
                 Spline(0.0), False, False),
            Tone(25600000, Spline(0.0),
                 pytest.approx(Spline(0.06644660940672625, -0.07601073581946988,
                                      0.02731215461568337, -0.0031192519636593086)),
                 Spline(0.0), False, False),
            Tone(25600000, Spline(0.0),
                 pytest.approx(Spline(0.014628776239280494, -0.03056855626504512,
                                      0.01740824258551721, -0.0014684625597525258)),
                 Spline(0.0), False, False),
            Tone(8, Spline(0.0), pytest.approx(Spline(0.0)), Spline(0.0), False, False)
        ],
        1: [Tone(409600008, Spline(0.0), Spline(1.0), Spline(0.0), False, False)]
    }

@with_rfsoc_params
def test_short_ramp_output1(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    ramp1 = StaticFunction()
    ramp2 = StaticFunction()
    s.add_step(200e-9) \
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
    assert action.float_value == 200e-9
    assert action.ramp is ramp2
    action = channels.channels[0].actions[1][1]
    assert action.cond
    assert not action.isramp
    assert not action.sync
    assert action.reloc_id == -1
    assert action.aid == 1
    assert action.tid == 1
    assert action.is_end
    assert action.seq_time == 200_000
    assert action.float_value == -1e-7
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
    assert action.float_value == 200e-9
    assert action.ramp is ramp1
    action = channels.channels[0].actions[2][1]
    assert action.cond
    assert not action.isramp
    assert not action.sync
    assert action.reloc_id == -1
    assert action.aid == 0
    assert action.tid == 1
    assert action.is_end
    assert action.seq_time == 200_000
    assert action.float_value == 0.0
    assert len(channels.channels[0].actions[3]) == 0 # ff

    comp.runtime_finalize(1)
    assert get_output() == {
        1: [Tone(41, Spline(0.0), pytest.approx(Spline(-2e-7, 5e-8), rel=3e-3),
                 pytest.approx(Spline(-4e-7 * np.pi, 1e-7 * np.pi), rel=3e-3),
                 False, False),
            Tone(41, Spline(0.0), pytest.approx(Spline(-1.5e-7, 5e-8), rel=3e-3),
                 pytest.approx(Spline(-3e-7 * np.pi, 1e-7 * np.pi), rel=3e-3),
                 False, False),
            Tone(8, Spline(0.0), Spline(0.0), pytest.approx(Spline(-2e-7 * np.pi)),
                 False, False)],
    }

@with_rfsoc_params
def test_short_ramp_output2(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    ramp1 = LinearRampNoSeg(0.1, 0.2)
    ramp2 = LinearRamp(0.1, 0.2)
    s.add_step(5e-9) \
      .pulse('rfsoc/dds0/1/amp', ramp1) \
      .set('rfsoc/dds0/1/phase', ramp2) \
      .set('rfsoc/dds2/0/amp', 0.1)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        1: [Tone(4, Spline(0.0),
                 pytest.approx(Spline(0.1, 0.34296875, -1.332421875, 0.889453125),
                               rel=3e-3),
                 pytest.approx(Spline(0.6283185307179586, 1.5266195503332707,
                                      -0.8320358549228715, -0.06626516469243882),
                               rel=3e-3),
                 False, False),
            Tone(6, Spline(0.0), Spline(0.0), pytest.approx(Spline(0.4 * np.pi)),
                 False, False)],
        4: [Tone(10, Spline(0.0), Spline(0.1), Spline(0.0), False, False)],
    }

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
def test_rampfunc_error(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    st = s.add_step(0.01)
    def jiasd9f89asd():
        st.pulse('rfsoc/dds0/1/freq', ErrorSegment())
    jiasd9f89asd()
    comp.finalize()
    with pytest.raises(ValueError, match="JJJLLL---DFFDAFD") as exc:
        comp.runtime_finalize(1)
    check_bt(exc, max_bt, 'jiasd9f89asd')

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    st = s.add_step(0.01)
    def j98asdf():
        st.pulse('rfsoc/dds0/1/freq', ErrorEval())
    j98asdf()
    comp.finalize()
    with pytest.raises(ValueError, match="XXXJJFSJkdfFDSDF") as exc:
        comp.runtime_finalize(1)
    check_bt(exc, max_bt, 'j98asdf')

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

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.set('rfsoc/dds0/0/ff', True)
    s.add_step(0.01) \
      .pulse('rfsoc/dds1/1/freq', rtval.new_extern(lambda: 1.23)) \
      .pulse('rfsoc/dds2/0/ff', rtval.new_extern(lambda: True))
    # This causes a error to be thrown when converting to boolean
    def js89j308joro82qwe():
        s.set('rfsoc/dds0/0/ff', rtval.new_extern(lambda: np.array([1, 2])))
    js89j308joro82qwe()
    comp.finalize()
    with pytest.raises(ValueError) as exc:
        comp.runtime_finalize(1)
    check_bt(exc, max_bt, 'js89j308joro82qwe')

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.set('rfsoc/dds0/0/ff', True)
    s.add_step(0.01) \
      .pulse('rfsoc/dds1/1/freq', rtval.new_extern(lambda: 1.23)) \
      .pulse('rfsoc/dds2/0/ff', rtval.new_extern(lambda: True))
    # This causes a error to be thrown when converting to float
    def e083jafd():
        s.set('rfsoc/dds3/0/amp', rtval.new_extern(lambda: [1, 2]))
    e083jafd()
    comp.finalize()
    with pytest.raises(TypeError) as exc:
        comp.runtime_finalize(1)
    check_bt(exc, max_bt, 'e083jafd')

    def error_callback():
        raise ValueError("AAABBBCCC")

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.set('rfsoc/dds0/0/ff', True)
    s.add_step(0.01) \
      .pulse('rfsoc/dds1/1/freq', rtval.new_extern(lambda: 1.23)) \
      .pulse('rfsoc/dds2/0/ff', rtval.new_extern(lambda: True))
    def oqo8we9813fasd():
        s.set('rfsoc/dds0/0/ff', rtval.new_extern(error_callback))
    oqo8we9813fasd()
    comp.finalize()
    with pytest.raises(ValueError, match="AAABBBCCC") as exc:
        comp.runtime_finalize(1)
    check_bt(exc, max_bt, 'oqo8we9813fasd')

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.set('rfsoc/dds0/0/ff', True)
    s.add_step(0.01) \
      .pulse('rfsoc/dds1/1/freq', rtval.new_extern(lambda: 1.23)) \
      .pulse('rfsoc/dds2/0/ff', rtval.new_extern(lambda: True))
    def q9e8uasdfasd():
        s.set('rfsoc/dds3/0/amp', rtval.new_extern(error_callback))
    q9e8uasdfasd()
    comp.finalize()
    with pytest.raises(ValueError, match="AAABBBCCC") as exc:
        comp.runtime_finalize(1)
    check_bt(exc, max_bt, 'q9e8uasdfasd')

@with_rfsoc_params
def test_ff_output(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)

    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/ff', False)
    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/ff', True)
    s.add_step(1e-3) \
      .pulse('rfsoc/dds0/0/ff', True)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        0: [Tone(409600, Spline(0.0), Spline(0.0), Spline(0.0), False, False),
            Tone(819208, Spline(0.0), Spline(0.0), Spline(0.0), False, True)],
    }

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)

    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/ff', False)
    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/ff', True)
    s.add_step(1e-3) \
      .pulse('rfsoc/dds0/0/ff', True, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        0: [Tone(409600, Spline(0.0), Spline(0.0), Spline(0.0), False, False),
            Tone(409600, Spline(0.0), Spline(0.0), Spline(0.0), False, True),
            Tone(409608, Spline(0.0), Spline(0.0), Spline(0.0), True, True)],
    }

@with_rfsoc_params
def test_param_output(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)

    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/amp', 0)
    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/amp', 0.2)
    s.add_step(1e-3) \
      .pulse('rfsoc/dds0/0/amp', 0.2)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        0: [Tone(409600, Spline(0.0), Spline(0.0), Spline(0.0), False, False),
            Tone(819208, Spline(0.0), Spline(0.2), Spline(0.0), False, False)],
    }

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)

    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/amp', 0)
    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/amp', 0.2)
    s.add_step(1e-3) \
      .pulse('rfsoc/dds0/0/amp', 0.2, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        0: [Tone(409600, Spline(0.0), Spline(0.0), Spline(0.0), False, False),
            Tone(409600, Spline(0.0), Spline(0.2), Spline(0.0), False, False),
            Tone(409608, Spline(0.0), Spline(0.2), Spline(0.0), True, False)],
    }

@with_rfsoc_params
def test_sync_merge(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(0)
    s.set('rfsoc/dds0/0/freq', 120e6, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        0: [Tone(8, Spline(120e6), Spline(0.0), Spline(0.0), True, False)],
    }

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(100e-12)
    s.set('rfsoc/dds0/0/freq', 120e6, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        0: [Tone(8, Spline(120e6), Spline(0.0), Spline(0.0), True, False)],
    }

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(2e-9)
    s.set('rfsoc/dds0/0/freq', 120e6, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        0: [Tone(4, Spline(120e6), Spline(0.0), Spline(0.0), True, False),
            Tone(5, Spline(120e6), Spline(0.0), Spline(0.0), False, False)],
    }

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(0)
    s.set('rfsoc/dds0/0/freq', 120e6)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        0: [Tone(4, Spline(100e6, 110e6, -180e6, 90e6),
                 Spline(0.0), Spline(0.0), True, False),
            Tone(4, Spline(120e6), Spline(0.0), Spline(0.0), False, False)],
    }

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(100e-12)
    s.set('rfsoc/dds0/0/freq', 120e6)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        0: [Tone(4, Spline(100e6, 110e6, -180e6, 90e6),
                 Spline(0.0), Spline(0.0), True, False),
            Tone(4, Spline(120e6), Spline(0.0), Spline(0.0), False, False)],
    }

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(2e-9)
    s.set('rfsoc/dds0/0/freq', 120e6)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        0: [Tone(4, Spline(100e6, 110e6, -180e6, 90e6),
                 Spline(0.0), Spline(0.0), True, False),
            Tone(5, Spline(120e6), Spline(0.0), Spline(0.0), False, False)],
    }

@with_rfsoc_params
def test_dyn_seq1(max_bt):
    b1 = True
    v1 = 0.001

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.add_step(0.0005) \
      .pulse('rfsoc/dds0/0/amp', 0.2)
    s.conditional(rtval.new_extern(lambda: b1)) \
      .add_step(rtval.new_extern(lambda: v1)) \
      .set('rfsoc/dds0/0/amp', 0.1)
    s.set('rfsoc/dds0/0/amp', 0.5)
    s.set('rfsoc/dds0/1/amp', 0.1)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        0: [Tone(204800, Spline(0.0), Spline(0.2), Spline(0.0), False, False),
            Tone(409600, Spline(0.0), Spline(0.1), Spline(0.0), False, False),
            Tone(8, Spline(0.0), Spline(0.5), Spline(0.0), False, False)],
        1: [Tone(614400, Spline(0.0), Spline(0.0), Spline(0.0), False, False),
            Tone(8, Spline(0.0), Spline(0.1), Spline(0.0), False, False)],
    }

    b1 = False
    v1 = 0.001
    comp.runtime_finalize(2)
    assert get_output() == {
        0: [Tone(204800, Spline(0.0), Spline(0.2), Spline(0.0), False, False),
            Tone(8, Spline(0.0), Spline(0.5), Spline(0.0), False, False)],
        1: [Tone(204800, Spline(0.0), Spline(0.0), Spline(0.0), False, False),
            Tone(8, Spline(0.0), Spline(0.1), Spline(0.0), False, False)],
    }

    b1 = True
    v1 = 0
    comp.runtime_finalize(3)
    assert get_output() == {
        0: [Tone(204800, Spline(0.0), Spline(0.2), Spline(0.0), False, False),
            Tone(8, Spline(0.0), Spline(0.5), Spline(0.0), False, False)],
        1: [Tone(204800, Spline(0.0), Spline(0.0), Spline(0.0), False, False),
            Tone(8, Spline(0.0), Spline(0.1), Spline(0.0), False, False)],
    }

    b1 = False
    v1 = 0
    comp.runtime_finalize(4)
    assert get_output() == {
        0: [Tone(204800, Spline(0.0), Spline(0.2), Spline(0.0), False, False),
            Tone(8, Spline(0.0), Spline(0.5), Spline(0.0), False, False)],
        1: [Tone(204800, Spline(0.0), Spline(0.0), Spline(0.0), False, False),
            Tone(8, Spline(0.0), Spline(0.1), Spline(0.0), False, False)],
    }

@with_rfsoc_params
def test_tight_output1(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.add_step(2.441e-9) \
      .pulse('rfsoc/dds0/1/amp', 0.1) \
      .set('rfsoc/dds0/1/phase', 0.2) \
      .set('rfsoc/dds0/1/freq', 100e6) \
      .pulse('rfsoc/dds2/0/amp', 0.3)
    s.add_step(2.441e-9) \
      .set('rfsoc/dds0/1/phase', 0.3) \
      .set('rfsoc/dds0/1/freq', 120e6) \
      .set('rfsoc/dds0/1/ff', True) \
      .set('rfsoc/dds2/0/amp', 0.1)
    s.add_step(2.442e-9) \
      .set('rfsoc/dds0/1/amp', 0.9) \
      .set('rfsoc/dds0/1/phase', 0.1) \
      .set('rfsoc/dds0/1/freq', 200e6) \
      .set('rfsoc/dds2/0/ff', True) \
      .set('rfsoc/dds2/0/amp', 0.3)
    s.add_step(2.441e-9) \
      .pulse('rfsoc/dds0/1/amp', 0.2) \
      .pulse('rfsoc/dds0/1/phase', 0.2) \
      .set('rfsoc/dds0/1/ff', False) \
      .pulse('rfsoc/dds0/1/freq', 120e6) \
      .pulse('rfsoc/dds2/0/amp', 0.7)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        1: [Tone(4, pytest.approx(Spline(100e6, 110e6, -180e6, 90e6),
                                  rel=1e-5, abs=1e-3),
                 pytest.approx(Spline(0.1, 0.55, -0.9, 0.45)),
                 pytest.approx(Spline(0.4 * np.pi)),
                 False, False),
            Tone(8, Spline(200e6), Spline(0.9), pytest.approx(Spline(0.2 * np.pi)),
                 False, False)],
        4: [Tone(4, Spline(0.0),
                 pytest.approx(Spline(0.3, -0.8, 3, -1.8)), Spline(0.0),
                 False, True),
            Tone(8, Spline(0.0), Spline(0.3), Spline(0.0),
                 False, True)],
    }

@with_rfsoc_params
def test_tight_output2(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.add_step(2.441e-9) \
      .pulse('rfsoc/dds0/1/amp', 0.1, sync=True) \
      .set('rfsoc/dds0/1/phase', 0.2) \
      .set('rfsoc/dds0/1/freq', 100e6, sync=True) \
      .pulse('rfsoc/dds2/0/amp', 0.3)
    s.add_step(2.441e-9) \
      .set('rfsoc/dds0/1/phase', 0.3, sync=True) \
      .set('rfsoc/dds0/1/freq', 120e6) \
      .set('rfsoc/dds0/1/ff', True) \
      .set('rfsoc/dds2/0/amp', 0.1)
    s.add_step(2.442e-9) \
      .set('rfsoc/dds0/1/amp', 0.9) \
      .set('rfsoc/dds0/1/phase', 0.1) \
      .set('rfsoc/dds0/1/freq', 200e6, sync=True) \
      .set('rfsoc/dds2/0/ff', True) \
      .set('rfsoc/dds2/0/amp', 0.3)
    s.add_step(2.441e-9) \
      .pulse('rfsoc/dds0/1/amp', 0.2) \
      .pulse('rfsoc/dds0/1/phase', 0.2) \
      .set('rfsoc/dds0/1/ff', False) \
      .pulse('rfsoc/dds0/1/freq', 120e6) \
      .pulse('rfsoc/dds2/0/amp', 0.7)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        1: [Tone(4, pytest.approx(Spline(200e6, -320e6, 600e6, -360e6),
                                  rel=1e-5, abs=1e-3),
                 pytest.approx(Spline(0.1, 0.55, -0.9, 0.45)),
                 pytest.approx(Spline(0.4 * np.pi)),
                 True, False),
            Tone(8, Spline(200e6), Spline(0.9), pytest.approx(Spline(0.2 * np.pi)),
                 False, False)],
        4: [Tone(4, Spline(0.0),
                 pytest.approx(Spline(0.3, -0.8, 3, -1.8)), Spline(0.0),
                 False, True),
            Tone(8, Spline(0.0), Spline(0.3), Spline(0.0),
                 False, True)],
    }

@with_rfsoc_params
def test_tight_output3(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.add_step(2.441e-9) \
      .pulse('rfsoc/dds0/1/amp', 0.1) \
      .set('rfsoc/dds0/1/phase', 0.2) \
      .set('rfsoc/dds0/1/freq', 100e6) \
      .pulse('rfsoc/dds2/0/amp', 0.3)
    s.add_step(2.441e-9) \
      .set('rfsoc/dds0/1/phase', 0.3) \
      .set('rfsoc/dds0/1/freq', 120e6) \
      .set('rfsoc/dds0/1/ff', True, sync=True) \
      .set('rfsoc/dds2/0/amp', 0.1)
    s.add_step(2.442e-9) \
      .set('rfsoc/dds0/1/amp', 0.9) \
      .set('rfsoc/dds0/1/phase', 0.1) \
      .set('rfsoc/dds0/1/freq', 200e6) \
      .set('rfsoc/dds2/0/ff', True) \
      .set('rfsoc/dds2/0/amp', 0.3)
    s.add_step(2.441e-9) \
      .pulse('rfsoc/dds0/1/amp', 0.2) \
      .pulse('rfsoc/dds0/1/phase', 0.2) \
      .set('rfsoc/dds0/1/ff', False) \
      .pulse('rfsoc/dds0/1/freq', 120e6) \
      .pulse('rfsoc/dds2/0/amp', 0.7)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        1: [Tone(4, pytest.approx(Spline(120e6), rel=1e-5, abs=1e-3),
                 pytest.approx(Spline(0.1, 0.55, -0.9, 0.45)),
                 pytest.approx(Spline(0.4 * np.pi)),
                 True, False),
            Tone(8, Spline(200e6), Spline(0.9), pytest.approx(Spline(0.2 * np.pi)),
                 False, False)],
        4: [Tone(4, Spline(0.0),
                 pytest.approx(Spline(0.3, -0.8, 3, -1.8)), Spline(0.0),
                 False, True),
            Tone(8, Spline(0.0), Spline(0.3), Spline(0.0),
                 False, True)],
    }

@with_rfsoc_params
def test_tight_output4(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.add_step(2.441e-9) \
      .pulse('rfsoc/dds0/1/amp', 0.1) \
      .set('rfsoc/dds0/1/phase', 0.2) \
      .set('rfsoc/dds0/1/freq', 100e6, sync=True) \
      .pulse('rfsoc/dds2/0/amp', 0.3)
    s.add_step(2.441e-9) \
      .set('rfsoc/dds0/1/phase', 0.3) \
      .set('rfsoc/dds0/1/freq', 120e6) \
      .set('rfsoc/dds0/1/ff', True, sync=True) \
      .set('rfsoc/dds2/0/amp', 0.1)
    s.add_step(2.442e-9) \
      .set('rfsoc/dds0/1/amp', 0.9, sync=True) \
      .set('rfsoc/dds0/1/phase', 0.1) \
      .set('rfsoc/dds0/1/freq', 200e6) \
      .set('rfsoc/dds2/0/ff', True) \
      .set('rfsoc/dds2/0/amp', 0.3)
    s.add_step(2.441e-9) \
      .pulse('rfsoc/dds0/1/amp', 0.2) \
      .pulse('rfsoc/dds0/1/phase', 0.2) \
      .set('rfsoc/dds0/1/ff', False) \
      .pulse('rfsoc/dds0/1/freq', 120e6) \
      .pulse('rfsoc/dds2/0/amp', 0.7)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        1: [Tone(4, pytest.approx(Spline(200e6, -320e6, 600e6, -360e6),
                                  rel=1e-5, abs=1e-3),
                 pytest.approx(Spline(0.1, 0.55, -0.9, 0.45)),
                 pytest.approx(Spline(0.4 * np.pi)),
                 True, False),
            Tone(8, Spline(200e6), Spline(0.9), pytest.approx(Spline(0.2 * np.pi)),
                 False, False)],
        4: [Tone(4, Spline(0.0),
                 pytest.approx(Spline(0.3, -0.8, 3, -1.8)), Spline(0.0),
                 False, True),
            Tone(8, Spline(0.0), Spline(0.3), Spline(0.0),
                 False, True)],
    }

@with_rfsoc_params
def test_tight_output5(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.add_step(2.441e-9) \
      .pulse('rfsoc/dds0/1/amp', 0.1) \
      .set('rfsoc/dds0/1/phase', 0.2) \
      .set('rfsoc/dds0/1/freq', 100e6) \
      .pulse('rfsoc/dds2/0/amp', 0.3)
    s.add_step(2.441e-9) \
      .set('rfsoc/dds0/1/phase', 0.3) \
      .set('rfsoc/dds0/1/freq', 120e6) \
      .set('rfsoc/dds0/1/ff', True) \
      .set('rfsoc/dds2/0/amp', 0.1)
    s.add_step(2.442e-9) \
      .set('rfsoc/dds0/1/amp', 0.9) \
      .set('rfsoc/dds0/1/phase', 0.1, sync=True) \
      .set('rfsoc/dds0/1/freq', 200e6) \
      .set('rfsoc/dds2/0/ff', True) \
      .set('rfsoc/dds2/0/amp', 0.3)
    s.add_step(2.441e-9) \
      .pulse('rfsoc/dds0/1/amp', 0.2) \
      .pulse('rfsoc/dds0/1/phase', 0.2) \
      .set('rfsoc/dds0/1/ff', False) \
      .pulse('rfsoc/dds0/1/freq', 120e6) \
      .pulse('rfsoc/dds2/0/amp', 0.7)
    comp.finalize()

    comp.runtime_finalize(1)
    assert get_output() == {
        1: [Tone(4, pytest.approx(Spline(200e6, -320e6, 600e6, -360e6),
                                  rel=1e-5, abs=1e-3),
                 pytest.approx(Spline(0.1, 0.55, -0.9, 0.45)),
                 pytest.approx(Spline(0.4 * np.pi)),
                 True, False),
            Tone(8, Spline(200e6), Spline(0.9), pytest.approx(Spline(0.2 * np.pi)),
                 False, False)],
        4: [Tone(4, Spline(0.0),
                 pytest.approx(Spline(0.3, -0.8, 3, -1.8)), Spline(0.0),
                 False, True),
            Tone(8, Spline(0.0), Spline(0.3), Spline(0.0),
                 False, True)],
    }

@with_rfsoc_params
def test_dds_delay_rt_error(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    rb.set_dds_delay(0, rtval.new_extern(lambda: -0.001))
    comp.finalize()
    with pytest.raises(ValueError, match="DDS time offset -0.001 cannot be negative."):
        comp.runtime_finalize(1)

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    rb.set_dds_delay(1, rtval.new_extern(lambda: 1))
    comp.finalize()
    with pytest.raises(ValueError,
                       match="DDS time offset 1 cannot be more than 100ms."):
        comp.runtime_finalize(1)

@with_rfsoc_params
def test_dds_delay(max_bt):
    check_dds_delay(max_bt, False)
    check_dds_delay(max_bt, True)

def check_dds_delay(max_bt, use_rt):
    def wrap_value(v):
        if use_rt:
            return rtval.new_extern(lambda: v)
        return v
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    with pytest.raises(ValueError, match="DDS time offset -0.001 cannot be negative."):
        rb.set_dds_delay(0, -0.001)
    with pytest.raises(ValueError,
                       match="DDS time offset 1 cannot be more than 100ms."):
        rb.set_dds_delay(1, 1)

    rb.set_dds_delay(1, wrap_value(1e-3))
    rb.set_dds_delay(0, wrap_value(1e-6))
    s.add_step(1e-3) \
      .pulse('rfsoc/dds0/0/amp', 0.1) \
      .set('rfsoc/dds0/1/phase', 0.2) \
      .set('rfsoc/dds1/1/freq', 100e6) \
      .pulse('rfsoc/dds2/0/amp', 0.3)
    comp.finalize()

    if not use_rt:
        channels = get_channel_info(rb, s)
        assert channels.dds_delay == {
            0: 1000_000,
            1: 1000_000_000
        }

    comp.runtime_finalize(1)
    if use_rt:
        channels = get_channel_info(rb, s)
        assert channels.dds_delay == {
            0: 1000_000,
            1: 1000_000_000
        }
    assert get_output() == {
        0: [Tone(410, Spline(0.0), Spline(0.0), Spline(0.0), False, False),
            Tone(409600, Spline(0.0), Spline(0.1), Spline(0.0), False, False),
            Tone(409198, Spline(0.0), Spline(0.0), Spline(0.0), False, False)],
        1: [Tone(410, Spline(0.0), Spline(0.0), Spline(0.0), False, False),
            Tone(818798, Spline(0.0), Spline(0.0), Spline(0.4 * np.pi), False, False)],
        3: [Tone(409600, Spline(0.0), Spline(0.0), Spline(0.0), False, False),
            Tone(409608, Spline(100e6), Spline(0.0), Spline(0.0), False, False)],
        4: [Tone(409600, Spline(0.0), Spline(0.3), Spline(0.0), False, False),
            Tone(409608, Spline(0.0), Spline(0.0), Spline(0.0), False, False)],
    }

@with_rfsoc_params
def test_cond_ramp_error(max_bt):
    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.conditional(False).add_step(1) \
      .set('rfsoc/dds0/0/amp', ErrorEval()) \
      .pulse('rfsoc/dds0/1/amp', ErrorEval())
    comp.finalize()
    comp.runtime_finalize(1)

    s, comp = new_seq_compiler(max_bt)
    rb = add_rfsoc_backend(comp)
    s.conditional(rtval.new_extern(lambda: False)) \
      .add_step(rtval.new_extern(lambda: 0)) \
      .set('rfsoc/dds0/0/amp', DivLengthFunction()) \
      .pulse('rfsoc/dds0/1/amp', DivLengthFunction())
    comp.finalize()
    comp.runtime_finalize(1)
