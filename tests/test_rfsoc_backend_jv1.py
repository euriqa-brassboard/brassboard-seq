#

from brassboard_seq.action import Blackman, BlackmanSquare, LinearRamp, \
     RampFunction, _RampFunctionBase, SeqCubicSpline
from brassboard_seq.config import Config
from brassboard_seq import backend, rtval, seq
from brassboard_seq.rfsoc_backend import Jaqal_v1, Jaqalv1Generator, RFSOCBackend

import brassboard_seq_test_utils as test_utils
import brassboard_seq_rfsoc_backend_utils as rfsoc_utils

import pytest
import re
import random
import itertools
import numpy as np
import copy

global_conf = Config()
global_conf.add_supported_prefix('artiq')
global_conf.add_supported_prefix('rfsoc')

class ParamPulses:
    def __init__(self):
        self.cycles = 0
        self.pulses = []

    def add(self, d):
        cycles = d['cycles']
        assert (self.cycles == 0) == d['trig']
        if d['param'] == 'frame_rot':
            assert d['fwd'] == 0
            assert d['inv'] == 0
            assert not d['eof']
            assert not d['clr']
            assert d['spline'] == [0.0, 0.0, 0.0, 0.0]
        else:
            assert not d['enable']
            if d['param'] != 'freq':
                assert not d['sync']
                assert not d['ff']
            self.pulses.append({'cycles': cycles, 'spline': d['spline'],
                                'sync': d['sync'], 'ff': d['ff']})
        self.cycles += cycles

class TonePulses:
    def __init__(self):
        self.freq = ParamPulses()
        self.amp = ParamPulses()
        self.phase = ParamPulses()
        self.frame_rot = ParamPulses()

    def add(self, d):
        param = d['param']
        if param == 'freq':
            self.freq.add(d)
        elif param == 'amp':
            self.amp.add(d)
        elif param == 'phase':
            self.phase.add(d)
        elif param == 'frame_rot':
            self.frame_rot.add(d)
        else:
            assert False

    def get_cycles(self):
        cycles1 = self.freq.cycles
        cycles2 = self.amp.cycles
        cycles3 = self.phase.cycles
        cycles4 = self.frame_rot.cycles
        assert cycles1 == cycles2
        assert cycles1 == cycles3
        assert cycles1 == cycles4
        return cycles1

    def to_dict(self):
        return dict(freq=self.freq.pulses, amp=self.amp.pulses, phase=self.phase.pulses)

class CompilerTester:
    def __init__(self):
        self.gen = Jaqalv1Generator()
    def new_env(self, *args):
        s = seq.Seq(global_conf, *args)
        comp = backend.SeqCompiler(s)
        rb = RFSOCBackend(self.gen)
        comp.add_backend('rfsoc', rb)
        comp.add_backend('artiq', backend.Backend()) # Dummy backend
        return s, comp, rb

    def get_output(self):
        pulses = [TonePulses() for _ in range(64)]
        for board_id in range(4):
            prefix = self.gen.get_prefix(board_id)
            sequence = self.gen.get_sequence(board_id)
            for inst in Jaqal_v1.extract_pulses(prefix + sequence):
                d = inst.to_dict()
                assert d['type'] == 'pulse_data'
                chn = d['channel']
                tone = d['tone']
                assert chn >= 0 and chn < 8
                assert tone == 0 or tone == 1
                chn_pulses = pulses[(board_id * 8 + chn) * 2 + tone]
                chn_pulses.add(d)

        stripped_pulses = {}
        total_cycles = 0
        for chn in range(32):
            chn_pulses1 = pulses[chn * 2]
            chn_pulses2 = pulses[chn * 2 + 1]

            chn_cycles1 = chn_pulses1.get_cycles()
            chn_cycles2 = chn_pulses2.get_cycles()
            assert chn_cycles1 == chn_cycles2
            if chn_cycles1 == 0:
                continue
            if total_cycles == 0:
                total_cycles = chn_cycles1
            else:
                assert total_cycles == chn_cycles1
            stripped_pulses[chn * 2] = chn_pulses1.to_dict()
            stripped_pulses[chn * 2 + 1] = chn_pulses2.to_dict()

        return stripped_pulses

    def check_output(self, expected):
        pulses = self.get_output()
        new_expected = {}
        for (k, v) in expected.items():
            newv = {}
            new_expected[k] = newv
            for (param, ps) in v.items():
                newv[param] = approx_pulses(param, ps)
        assert pulses == new_expected

gentest = CompilerTester()

def Spline(order0=0.0, order1=0.0, order2=0.0, order3=0.0):
    return [order0, order1, order2, order3]

def Pulse(cycles, spline=Spline(), sync=False, ff=False):
    return dict(cycles=cycles, spline=spline, sync=sync, ff=ff)

def approx_pulses(param, ps):
    if param == 'freq':
        _abs = 0.0015
    elif param == 'amp':
        _abs = 1.5e-5
    elif param == 'phase':
        _abs = 2e-12
    else:
        assert False
    def approx_pulse(p):
        p = copy.copy(p)
        p['spline'] = pytest.approx(p['spline'], abs=_abs, rel=1e-9)
        return p
    return [approx_pulse(p) for p in ps]

def check_bt(exc, max_bt, *names):
    fnames = [tb.name for tb in exc.traceback]
    for name in names:
        if max_bt == 0:
            assert name not in fnames
        else:
            assert name in fnames

with_rfsoc_params = pytest.mark.parametrize("max_bt", [0, 5, 500])

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
                isramp = isinstance(action_value, _RampFunctionBase)
                if rfsoc_action.tid == action_info['tid']:
                    assert not seen[0]
                    seen[0] = True
                    value = action_value
                    assert isramp == rfsoc_action.isramp
                else:
                    assert rfsoc_action.tid == action_info['end_tid']
                    isramp = isinstance(test_utils.action_get_value(action),
                                        _RampFunctionBase)
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
        isramp = isinstance(action_value, _RampFunctionBase)
        if test_utils.action_get_is_pulse(action) or isramp:
            assert seen[1]
        else:
            assert not seen[1]
    assert all(bool_values_used)
    assert all(float_values_used)
    assert all(relocations_used)

    return channels

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

class CustomSegment(RampFunction):
    def __init__(self, seg):
        self.seg = seg
        super().__init__()

    def eval(self, t, length, oldval):
        return t

    def spline_segments(self, length, oldval):
        return self.seg

class DivLengthFunction(RampFunction):
    def __init__(self):
        super().__init__()

    def eval(self, t, length, oldval):
        return t / length

@with_rfsoc_params
def test_channels(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    assert not rb.has_output
    comp.finalize()
    assert not rb.has_output

    s, comp, rb = gentest.new_env(max_bt)
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
    assert chn_ids == [2, 3, 4, 0, 1, 5]
    assert channels.chn_map == {1: (0, 'amp'), 2: (1, 'freq'),
                                4: (2, 'phase'), 5: (3, 'ff'),
                                6: (1, 'amp'), 7: (1, 'ff'),
                                8: (3, 'freq')}
    comp.runtime_finalize(1)
    with pytest.raises(IndexError, match="Board index should be in \\[0, 3\\]"):
        gentest.gen.get_prefix(-1)
    with pytest.raises(IndexError, match="Board index should be in \\[0, 3\\]"):
        gentest.gen.get_prefix(4)
    with pytest.raises(IndexError, match="Board index should be in \\[0, 3\\]"):
        gentest.gen.get_sequence(-1)
    with pytest.raises(IndexError, match="Board index should be in \\[0, 3\\]"):
        gentest.gen.get_sequence(4)
    gentest.check_output({
        0: {
          'freq': [Pulse(8)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
        1: {
          'freq': [Pulse(8)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
        2: {
          'freq': [Pulse(8)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
        3: {
          'freq': [Pulse(8)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
        4: {
          'freq': [Pulse(8)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
        5: {
          'freq': [Pulse(8)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
    })

    s, comp, rb = gentest.new_env(max_bt)
    assert s.get_channel_id('rfsoc/dds0/0') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds0/0'):
        comp.finalize()

    s, comp, rb = gentest.new_env(max_bt)
    assert s.get_channel_id('rfsoc/dds0/0/freq/a') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds0/0/freq/a'):
        comp.finalize()

    s, comp, rb = gentest.new_env(max_bt)
    assert s.get_channel_id('rfsoc/ch0/0/freq') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/ch0/0/freq'):
        comp.finalize()

    s, comp, rb = gentest.new_env(max_bt)
    assert s.get_channel_id('rfsoc/dds50/0/freq') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds50/0/freq'):
        comp.finalize()

    s, comp, rb = gentest.new_env(max_bt)
    assert s.get_channel_id('rfsoc/dds10/2/freq') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds10/2/freq'):
        comp.finalize()

    s, comp, rb = gentest.new_env(max_bt)
    assert s.get_channel_id('rfsoc/dds10/0/param') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds10/0/param'):
        comp.finalize()

@with_rfsoc_params
def test_output1(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    s.add_step(0.01) \
      .pulse('rfsoc/dds0/1/amp', 0.2) \
      .set('rfsoc/dds0/1/freq', 100e6) \
      .set('rfsoc/dds0/1/phase', 0.1) \
      .set('rfsoc/dds0/1/ff', True)
    comp.finalize()
    channels = get_channel_info(rb, s)
    assert len(channels.channels) == 2
    assert channels.channels[0].chn == 1
    assert channels.channels[1].chn == 0
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
    gentest.check_output({
        0: {
          'freq': [Pulse(4096008)],
          'amp': [Pulse(4096008)],
          'phase': [Pulse(4096008)]
        },
        1: {
          'freq': [Pulse(4096000, Spline(100e6), ff=True),
                   Pulse(8, Spline(100e6), ff=True)],
          'amp': [Pulse(4096000, Spline(0.2)), Pulse(8)],
          'phase': [Pulse(4096000, Spline(0.1)), Pulse(8, Spline(0.1))]
        },
    })

@with_rfsoc_params
def test_output2(max_bt):
    b1 = True
    s, comp, rb = gentest.new_env(max_bt)
    s.conditional(rtval.new_extern(lambda: b1)) \
      .add_step(rtval.new_extern(lambda: 0.01)) \
      .pulse('rfsoc/dds0/1/amp', rtval.new_extern(lambda: 0.2)) \
      .set('rfsoc/dds0/1/freq', rtval.new_extern(lambda: 100e6), sync=True) \
      .set('rfsoc/dds0/1/phase', rtval.new_extern(lambda: 0.1)) \
      .set('rfsoc/dds0/1/ff', rtval.new_extern(lambda: True), sync=False)
    comp.finalize()
    channels = get_channel_info(rb, s)
    assert len(channels.channels) == 2
    assert channels.channels[0].chn == 1
    assert channels.channels[1].chn == 0
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
    gentest.check_output({
        0: {
          'freq': [Pulse(4096008)],
          'amp': [Pulse(4096008)],
          'phase': [Pulse(4096008)]
        },
        1: {
          'freq': [Pulse(4096000, Spline(100e6), sync=True, ff=True),
                   Pulse(8, Spline(100e6), ff=True)],
          'amp': [Pulse(4096000, Spline(0.2)), Pulse(8)],
          'phase': [Pulse(4096000, Spline(0.1)), Pulse(8, Spline(0.1))]
        },
    })

    b1 = False
    comp.runtime_finalize(2)
    gentest.check_output({
        0: {
          'freq': [Pulse(8)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
        1: {
          'freq': [Pulse(8)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
    })

@with_rfsoc_params
def test_output3(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    s.conditional(False) \
      .add_step(rtval.new_extern(lambda: 0.01)) \
      .pulse('rfsoc/dds0/1/amp', rtval.new_extern(lambda: 0.2)) \
      .set('rfsoc/dds0/1/freq', rtval.new_extern(lambda: 100e6), sync=True) \
      .set('rfsoc/dds0/1/phase', rtval.new_extern(lambda: 0.1)) \
      .set('rfsoc/dds0/1/ff', rtval.new_extern(lambda: True), sync=False)
    comp.finalize()
    channels = get_channel_info(rb, s)
    assert len(channels.channels) == 2
    assert channels.channels[0].chn == 1
    assert channels.channels[1].chn == 0
    assert len(channels.channels[0].actions[0]) == 0 # freq
    assert len(channels.channels[0].actions[1]) == 0 # phase
    assert len(channels.channels[0].actions[2]) == 0 # amp
    assert len(channels.channels[0].actions[3]) == 0 # ff

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(8)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
        1: {
          'freq': [Pulse(8)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
    })

@with_rfsoc_params
def test_ramp_output1(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    ramp1 = StaticFunction()
    ramp2 = StaticFunction()
    s.add_step(0.01) \
      .pulse('rfsoc/dds0/1/amp', ramp1) \
      .set('rfsoc/dds0/1/phase', ramp2)
    comp.finalize()
    channels = get_channel_info(rb, s)
    assert len(channels.channels) == 2
    assert channels.channels[0].chn == 1
    assert channels.channels[1].chn == 0
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
    gentest.check_output({
        0: {
          'freq': [Pulse(4096008)],
          'amp': [Pulse(4096008)],
          'phase': [Pulse(4096008)]
        },
        1: {
          'freq': [Pulse(2048000), Pulse(2048000), Pulse(8)],
          'amp': [Pulse(2048000, Spline(-0.01, 0.0025)),
                  Pulse(2048000, Spline(-0.0075, 0.0025)),
                  Pulse(8)],
          'phase': [Pulse(2048000, Spline(-0.01, 0.0025)),
                    Pulse(2048000, Spline(-0.0075, 0.0025)),
                    Pulse(8, Spline(-0.005))]
        },
    })

@with_rfsoc_params
def test_ramp_output2(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    ramp1 = StaticFunction()
    ramp2 = StaticFunction()
    s.add_step(rtval.new_extern(lambda: 0.01)) \
      .pulse('rfsoc/dds0/1/amp', ramp1) \
      .set('rfsoc/dds0/1/phase', ramp2)
    comp.finalize()
    channels = get_channel_info(rb, s)
    assert len(channels.channels) == 2
    assert channels.channels[0].chn == 1
    assert channels.channels[1].chn == 0
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
    gentest.check_output({
        0: {
          'freq': [Pulse(4096008)],
          'amp': [Pulse(4096008)],
          'phase': [Pulse(4096008)]
        },
        1: {
          'freq': [Pulse(2048000), Pulse(2048000), Pulse(8)],
          'amp': [Pulse(2048000, Spline(-0.01, 0.0025)),
                  Pulse(2048000, Spline(-0.0075, 0.0025)),
                  Pulse(8)],
          'phase': [Pulse(2048000, Spline(-0.01, 0.0025)),
                    Pulse(2048000, Spline(-0.0075, 0.0025)),
                    Pulse(8, Spline(-0.005))]
        },
    })

@with_rfsoc_params
def test_ramp_output3(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
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
    gentest.check_output({
        0: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048008)],
          'phase': [Pulse(2048008)]
        },
        1: {
          'freq': [Pulse(1024000), Pulse(1024000), Pulse(8)],
          'amp': [Pulse(1024000, Spline(0.1, 0.05)),
                  Pulse(1024000, Spline(0.15, 0.05)),
                  Pulse(8)],
          'phase': [Pulse(1024000), Pulse(1024000), Pulse(8)]
        },
        4: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048008, Spline(0.1))],
          'phase': [Pulse(2048008)]
        },
        5: {
          'freq': [Pulse(2048000), Pulse(8)],
          'amp': [Pulse(2048000), Pulse(8)],
          'phase': [Pulse(2048000, Spline(0.1, 0.1)), Pulse(8, Spline(0.2))]
        },
        8: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048008)],
          'phase': [Pulse(2048008)]
        },
        9: {
          'freq': [Pulse(2048000), Pulse(8)],
          'amp': [Pulse(2048000, Spline(0.2, 0.1, 0.9, 0.3)),
                  Pulse(8, pytest.approx(Spline(-0.5), abs=4e-5))],
          'phase': [Pulse(2048000), Pulse(8)]
        },
    })

@with_rfsoc_params
def test_ramp_output4(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    ramp1 = LinearRampNoSeg(0.1, 0.2)
    ramp2 = SeqCubicSpline(0.2, 0.1, 0.9, 0.3)
    s.add_step(5e-3) \
      .pulse('rfsoc/dds0/1/phase', ramp1) \
      .set('rfsoc/dds2/1/phase', ramp2) \
      .set('rfsoc/dds2/0/amp', 0.1) \
      .set('rfsoc/dds2/0/phase', -1 / 2 / np.pi)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048008)],
          'phase': [Pulse(2048008)]
        },
        1: {
          'freq': [Pulse(1024000), Pulse(1024000), Pulse(8)],
          'amp': [Pulse(1024000), Pulse(1024000), Pulse(8)],
          'phase': [Pulse(1024000, Spline(0.1, 0.05)),
                    Pulse(1024000, Spline(0.15, 0.05)),
                    Pulse(8)]
        },
        4: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048008, Spline(0.1))],
          'phase': [Pulse(2048008, Spline(-1.0 / 2 / np.pi))]
        },
        5: {
          'freq': [Pulse(2048000), Pulse(8)],
          'amp': [Pulse(2048000), Pulse(8)],
          'phase': [Pulse(2048000, Spline(0.2, 0.1, 0.9, 0.3)),
                    Pulse(8, Spline(-0.5))]
        },
    })

@with_rfsoc_params
def test_ramp_output5(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
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
    gentest.check_output({
        0: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048008, Spline(0.2))],
          'phase': [Pulse(2048008, Spline(-0.5 / 2 / np.pi))]
        },
        1: {
          'freq': [Pulse(512000), Pulse(1024000), Pulse(512000), Pulse(8)],
          'amp': [Pulse(512000, Spline(0.2, 0.3)),
                  Pulse(1024000, Spline(0.5, -0.2)),
                  Pulse(512000, Spline(0.3, -0.1)),
                  Pulse(8, Spline(0.2))],
          'phase': [Pulse(512000, Spline(-0.1, -0.2)),
                    Pulse(1024000, Spline(-0.3, -0.4)),
                    Pulse(512000, Spline(0.3, 0.6)),
                    Pulse(8, Spline(-0.1))]
        },
        4: {
          'freq': [Pulse(1024000, Spline(80e6, 20e6)),
                   Pulse(1024000, Spline(100e6, -20e6)),
                   Pulse(8, Spline(80e6))],
          'amp': [Pulse(1024000), Pulse(1024000), Pulse(8)],
          'phase': [Pulse(1024000), Pulse(1024000), Pulse(8)]
        },
        5: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048008)],
          'phase': [Pulse(2048008)]
        },
    })

@with_rfsoc_params
def test_ramp_output6(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
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
    gentest.check_output({
        0: {
          'freq': [
            Pulse(409600),
            # freq 0.0
            Pulse(409600, Spline(100e6, 2.5e6, -1.25e6, 0.3125e6)),
            # freq 0.1
            Pulse(409600, Spline(101.5625e6, 0.9375e6, -0.3125e6, 0.3125e6)),
            # freq 0.2
            Pulse(409600, Spline(102.5e6, 1.25e6, 0.625e6, 0.3125e6), ff=True),
            # freq 0.3
            Pulse(409600, Spline(104.6875e6, 3.4375e6, 1.5625e6, 0.3125e6), ff=True),
            # freq 1.0
            Pulse(409600, Spline(10e6, 25e6, -1.875e6, -0.3125e6), ff=True),
            # freq 1.1
            Pulse(409600, Spline(32.8125e6, 20.3125e6, -2.8125e6, -0.3125e6), ff=True),
            # freq 1.2
            Pulse(409600, Spline(50e6, 13.75e6, -3.75e6, -0.3125e6)),
            # freq 1.3
            Pulse(409600, Spline(59.6875e6, 5.3125e6, -4.6875e6, -0.3125e6)),
            Pulse(409600, Spline(60e6)),
            Pulse(409608, Spline(60e6))],
          'amp': [
            # amp 0.0
            Pulse(409600, Spline(0.2, -0.025, 0.05625, -0.0046875)),
            # amp 0.1
            Pulse(409600, Spline(0.2265625, 0.0734375, 0.0421875, -0.0046875)),
            # amp 0.2
            Pulse(409600, Spline(0.3375, 0.14375, 0.028125, -0.0046875)),
            # amp 0.3
            Pulse(409600, Spline(0.5046875, 0.1859375, 0.0140625, -0.0046875)),
            # amp 1.0
            Pulse(409600, Spline(0.5, -0.025, 0.03125, -0.003125)),
            # amp 1.1
            Pulse(409600, Spline(0.503125, 0.028125, 0.021875, -0.003125)),
            # amp 1.2
            Pulse(409600, Spline(0.55, 0.0625, 0.0125, -0.003125)),
            # amp 1.3
            Pulse(409600, Spline(0.621875, 0.078125, 0.003125, -0.003125)),
            Pulse(409600, Spline(0.7)),
            Pulse(409600, Spline(0.7)),
            Pulse(409608, Spline(0.7))
          ],
          'phase': [
            Pulse(409600),
            Pulse(409600),
            # phase 0.0
            Pulse(409600, Spline(0.4, -0.05, 0.00625, -0.003125)),
            # phase 0.1
            Pulse(409600, Spline(0.353125, -0.046875, -0.003125, -0.003125)),
            # phase 0.2
            Pulse(409600, Spline(0.3, -0.0625, -0.0125, -0.003125)),
            # phase 0.3
            Pulse(409600, Spline(0.221875, -0.096875, -0.021875, -0.003125)),
            # phase 1.0
            Pulse(409600, Spline(0.2, 0.175, 0.00625, -0.0140625)),
            # phase 1.1
            Pulse(409600, Spline(0.3671875, 0.1453125, -0.0359375, -0.0140625)),
            # phase 1.2
            Pulse(409600, Spline(0.4625, 0.03125, -0.078125, -0.0140625)),
            # phase 1.3
            Pulse(409600, Spline(0.4015625, -0.1671875, -0.1203125, -0.0140625)),
            Pulse(409608, Spline(0.1))
            ]
        },
        1: {
          'freq': [Pulse(4505608)],
          'amp': [Pulse(4505608)],
          'phase': [Pulse(4505608)]
        },
        4: {
          'freq': [Pulse(409600), Pulse(4096008)],
          'amp': [Pulse(409600), Pulse(4096008, Spline(0.1))],
          'phase': [Pulse(409600), Pulse(4096008)]
        },
        5: {
          'freq': [Pulse(4505608)],
          'amp': [Pulse(4505608)],
          'phase': [Pulse(4505608)]
        },
    })

@with_rfsoc_params
def test_ramp_output7(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    s.add_step(1) \
      .set('rfsoc/dds0/0/amp', Blackman(1.0)) \
      .set('rfsoc/dds0/1/amp', 1.0)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(8)
          ],
          'amp': [
            Pulse(25600000, Spline(0.0, 0.00015745877326822644,
                                   0.01300285490625988, 0.001468462559752276)),
            Pulse(25600000, Spline(0.014628776239280439, 0.03074418247908131,
                                   0.01795439872470489, 0.003119251963659586)),
            Pulse(25600000, Spline(0.06644660940672625, 0.07585724657247933,
                                   0.028056006241789055, 0.0017298791015365467)),
            Pulse(25600000, Spline(0.17208974132253135, 0.13669965899911896,
                                   0.033698736281407626, -0.002488136603058244)),
            Pulse(25600000, Spline(0.34, 0.19610073763144964,
                                   0.026062367723541868, -0.007389931667371563)),
            Pulse(25600000, Spline(0.5547731736876211, 0.22576257627864393,
                                   0.003121741649712906, -0.010104101022703649)),
            Pulse(25600000, Spline(0.7735533905932739, 0.20184724575694735,
                                   -0.028177599439071166, -0.008714728160579277)),
            Pulse(25600000, Spline(0.9385083087505672, 0.11992457534234635,
                                   -0.05499955158835057, -0.0034333325045636798)),
            Pulse(25600000, Spline(1.0, 0.00037452534804338633,
                                   -0.06529954910203895, 0.003433332504564568)),
            Pulse(25600000, Spline(0.9385083087505672, -0.11934786239705719,
                                   -0.05432178392081566, 0.008714728160582386)),
            Pulse(25600000, Spline(0.7735533905932739, -0.2016937565099568,
                                   -0.027190561418399817, 0.010104101022704093)),
            Pulse(25600000, Spline(0.5547731736876214, -0.22605567807641802,
                                   0.0038925727214200734, 0.007389931667376892)),
            Pulse(25600000, Spline(0.34, -0.19663272175275923,
                                   0.026234326472228675, 0.002488136603061575)),
            Pulse(25600000, Spline(0.17208974132253121, -0.13715889636066686,
                                   0.03324564354639803, -0.0017298791015359916)),
            Pulse(25600000, Spline(0.06644660940672625, -0.07601073581946988,
                                   0.02731215461568337, -0.0031192519636593086)),
            Pulse(25600000, Spline(0.014628776239280494, -0.03056855626504512,
                                   0.01740824258551721, -0.0014684625597525258)),
            Pulse(8)
          ],
          'phase': [
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(8)
          ]
        },
        1: {
          'freq': [Pulse(409600008)],
          'amp': [Pulse(409600008, Spline(1.0))],
          'phase': [Pulse(409600008)]
        }
    })

@with_rfsoc_params
def test_ramp_reuse_spline(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    r = SeqCubicSpline(0.1, 0.4, -0.2, 0.1)
    s.add_step(5e-3) \
      .set('rfsoc/dds0/0/amp', r)
    s.add_step(10e-3) \
      .set('rfsoc/dds0/0/amp', r)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
            'freq': [Pulse(2048000), Pulse(4096000), Pulse(8)],
            'amp': [Pulse(2048000, Spline(0.1, 0.4, -0.2, 0.1)),
                    Pulse(4096000, Spline(0.1, 0.4, -0.2, 0.1)),
                    Pulse(8, Spline(0.4))],
            'phase': [Pulse(2048000), Pulse(4096000), Pulse(8)],
        },
        1: {
            'freq': [Pulse(6144008)],
            'amp': [Pulse(6144008)],
            'phase': [Pulse(6144008)],
        },
    })

@with_rfsoc_params
def test_ramp_reuse_oldval(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    r = RampUpAndDown(0.5, 0.9)
    s.add_step(5e-3) \
      .set('rfsoc/dds0/0/amp', r)
    s.set('rfsoc/dds0/0/amp', 0.3)
    s.add_step(10e-3) \
      .set('rfsoc/dds0/0/amp', r)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
            'freq': [Pulse(1024000), Pulse(1024000), Pulse(2048000),
                     Pulse(2048000), Pulse(8)],
            'amp': [Pulse(1024000, Spline(0, 0.9)), Pulse(1024000, Spline(0.9, -0.9)),
                    Pulse(2048000, Spline(0.3, 0.6)), Pulse(2048000, Spline(0.9, -0.6)),
                    Pulse(8, Spline(0.3))],
            'phase': [Pulse(1024000), Pulse(1024000), Pulse(2048000),
                     Pulse(2048000), Pulse(8)],
        },
        1: {
            'freq': [Pulse(6144008)],
            'amp': [Pulse(6144008)],
            'phase': [Pulse(6144008)],
        },
    })

@with_rfsoc_params
def test_ramp_reuse_linear(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    r = LinearRamp(0.1, 0.9)
    r2 = LinearRampNoSeg(0.1, 0.9)
    s.add_step(5e-3) \
      .set('rfsoc/dds0/0/amp', r) \
      .set('rfsoc/dds0/1/amp', r2)
    s.add_step(10e-3) \
      .set('rfsoc/dds0/0/amp', r2) \
      .set('rfsoc/dds0/1/amp', r)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
            'freq': [Pulse(2048000), Pulse(2048000), Pulse(2048000), Pulse(8)],
            'amp': [Pulse(2048000, Spline(0.1, 0.8)), Pulse(2048000, Spline(0.1, 0.4)),
                    Pulse(2048000, Spline(0.5, 0.4)), Pulse(8, Spline(0.9))],
            'phase': [Pulse(2048000), Pulse(2048000), Pulse(2048000), Pulse(8)],
        },
        1: {
            'freq': [Pulse(1024000), Pulse(1024000), Pulse(4096000), Pulse(8)],
            'amp': [Pulse(1024000, Spline(0.1, 0.4)), Pulse(1024000, Spline(0.5, 0.4)),
                    Pulse(4096000, Spline(0.1, 0.8)), Pulse(8, Spline(0.9))],
            'phase': [Pulse(1024000), Pulse(1024000), Pulse(4096000), Pulse(8)],
        },
    })

@with_rfsoc_params
def test_ramp_reuse_blackman(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    r = Blackman(1.0)
    s.add_step(1) \
      .set('rfsoc/dds0/0/amp', r) \
      .set('rfsoc/dds0/1/amp', 1.0)
    s.add_step(2) \
      .set('rfsoc/dds0/0/amp', r) \
      .set('rfsoc/dds0/1/amp', 1.0)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(8)
          ],
          'amp': [
            Pulse(25600000, Spline(0.0, 0.00015745877326822644,
                                   0.01300285490625988, 0.001468462559752276)),
            Pulse(25600000, Spline(0.014628776239280439, 0.03074418247908131,
                                   0.01795439872470489, 0.003119251963659586)),
            Pulse(25600000, Spline(0.06644660940672625, 0.07585724657247933,
                                   0.028056006241789055, 0.0017298791015365467)),
            Pulse(25600000, Spline(0.17208974132253135, 0.13669965899911896,
                                   0.033698736281407626, -0.002488136603058244)),
            Pulse(25600000, Spline(0.34, 0.19610073763144964,
                                   0.026062367723541868, -0.007389931667371563)),
            Pulse(25600000, Spline(0.5547731736876211, 0.22576257627864393,
                                   0.003121741649712906, -0.010104101022703649)),
            Pulse(25600000, Spline(0.7735533905932739, 0.20184724575694735,
                                   -0.028177599439071166, -0.008714728160579277)),
            Pulse(25600000, Spline(0.9385083087505672, 0.11992457534234635,
                                   -0.05499955158835057, -0.0034333325045636798)),
            Pulse(25600000, Spline(1.0, 0.00037452534804338633,
                                   -0.06529954910203895, 0.003433332504564568)),
            Pulse(25600000, Spline(0.9385083087505672, -0.11934786239705719,
                                   -0.05432178392081566, 0.008714728160582386)),
            Pulse(25600000, Spline(0.7735533905932739, -0.2016937565099568,
                                   -0.027190561418399817, 0.010104101022704093)),
            Pulse(25600000, Spline(0.5547731736876214, -0.22605567807641802,
                                   0.0038925727214200734, 0.007389931667376892)),
            Pulse(25600000, Spline(0.34, -0.19663272175275923,
                                   0.026234326472228675, 0.002488136603061575)),
            Pulse(25600000, Spline(0.17208974132253121, -0.13715889636066686,
                                   0.03324564354639803, -0.0017298791015359916)),
            Pulse(25600000, Spline(0.06644660940672625, -0.07601073581946988,
                                   0.02731215461568337, -0.0031192519636593086)),
            Pulse(25600000, Spline(0.014628776239280494, -0.03056855626504512,
                                   0.01740824258551721, -0.0014684625597525258)),
            Pulse(51200000, Spline(0.0, 0.00015745877326822644,
                                   0.01300285490625988, 0.001468462559752276)),
            Pulse(51200000, Spline(0.014628776239280439, 0.03074418247908131,
                                   0.01795439872470489, 0.003119251963659586)),
            Pulse(51200000, Spline(0.06644660940672625, 0.07585724657247933,
                                   0.028056006241789055, 0.0017298791015365467)),
            Pulse(51200000, Spline(0.17208974132253135, 0.13669965899911896,
                                   0.033698736281407626, -0.002488136603058244)),
            Pulse(51200000, Spline(0.34, 0.19610073763144964,
                                   0.026062367723541868, -0.007389931667371563)),
            Pulse(51200000, Spline(0.5547731736876211, 0.22576257627864393,
                                   0.003121741649712906, -0.010104101022703649)),
            Pulse(51200000, Spline(0.7735533905932739, 0.20184724575694735,
                                   -0.028177599439071166, -0.008714728160579277)),
            Pulse(51200000, Spline(0.9385083087505672, 0.11992457534234635,
                                   -0.05499955158835057, -0.0034333325045636798)),
            Pulse(51200000, Spline(1.0, 0.00037452534804338633,
                                   -0.06529954910203895, 0.003433332504564568)),
            Pulse(51200000, Spline(0.9385083087505672, -0.11934786239705719,
                                   -0.05432178392081566, 0.008714728160582386)),
            Pulse(51200000, Spline(0.7735533905932739, -0.2016937565099568,
                                   -0.027190561418399817, 0.010104101022704093)),
            Pulse(51200000, Spline(0.5547731736876214, -0.22605567807641802,
                                   0.0038925727214200734, 0.007389931667376892)),
            Pulse(51200000, Spline(0.34, -0.19663272175275923,
                                   0.026234326472228675, 0.002488136603061575)),
            Pulse(51200000, Spline(0.17208974132253121, -0.13715889636066686,
                                   0.03324564354639803, -0.0017298791015359916)),
            Pulse(51200000, Spline(0.06644660940672625, -0.07601073581946988,
                                   0.02731215461568337, -0.0031192519636593086)),
            Pulse(51200000, Spline(0.014628776239280494, -0.03056855626504512,
                                   0.01740824258551721, -0.0014684625597525258)),
            Pulse(8)
          ],
          'phase': [
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(25600000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(51200000),
            Pulse(8)
          ]
        },
        1: {
          'freq': [Pulse(1228800008)],
          'amp': [Pulse(1228800008, Spline(1.0))],
          'phase': [Pulse(1228800008)]
        }
    })

@with_rfsoc_params
def test_short_ramp_output1(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    ramp1 = StaticFunction()
    ramp2 = StaticFunction()
    s.add_step(200e-9) \
      .pulse('rfsoc/dds0/1/amp', ramp1) \
      .set('rfsoc/dds0/1/phase', ramp2)
    comp.finalize()
    channels = get_channel_info(rb, s)
    assert len(channels.channels) == 2
    assert channels.channels[0].chn == 1
    assert channels.channels[1].chn == 0
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
    gentest.check_output({
        0: {
          'freq': [Pulse(90)],
          'amp': [Pulse(90)],
          'phase': [Pulse(90)]
        },
        1: {
          'freq': [Pulse(41), Pulse(41), Pulse(8)],
          'amp': [Pulse(41, Spline(-2e-7, 5e-8)),
                  Pulse(41, Spline(-1.5e-7, 5e-8)),
                  Pulse(8)],
          'phase': [Pulse(41, Spline(-2e-7, 0.5e-7)),
                    Pulse(41, Spline(-1.5e-7, 0.5e-7)),
                    Pulse(8, Spline(-1e-7))]
        },
    })

@with_rfsoc_params
def test_short_ramp_output2(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    ramp1 = LinearRampNoSeg(0.1, 0.2)
    ramp2 = LinearRamp(0.1, 0.2)
    s.add_step(5e-9) \
      .pulse('rfsoc/dds0/1/amp', ramp1) \
      .set('rfsoc/dds0/1/phase', ramp2) \
      .set('rfsoc/dds2/0/amp', 0.1)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(10)],
          'amp': [Pulse(10)],
          'phase': [Pulse(10)]
        },
        1: {
          'freq': [Pulse(4), Pulse(6)],
          'amp': [Pulse(4, Spline(0.1, 0.35, -1.35, 0.9)), Pulse(6)],
          'phase': [Pulse(4, Spline(0.1, 0.25, -0.15)), Pulse(6, Spline(0.2))]
        },
        4: {
          'freq': [Pulse(10)],
          'amp': [Pulse(10, Spline(0.1))],
          'phase': [Pulse(10)]
        },
        5: {
          'freq': [Pulse(10)],
          'amp': [Pulse(10)],
          'phase': [Pulse(10)]
        },
    })

@with_rfsoc_params
def test_arg_error(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
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
    s, comp, rb = gentest.new_env(max_bt)
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
    s, comp, rb = gentest.new_env(max_bt)
    st = s.add_step(0.01)
    def jiasd9f89asd():
        st.pulse('rfsoc/dds0/1/freq', ErrorSegment())
    jiasd9f89asd()
    comp.finalize()
    with pytest.raises(ValueError, match="JJJLLL---DFFDAFD") as exc:
        comp.runtime_finalize(1)
    check_bt(exc, max_bt, 'jiasd9f89asd')

    s, comp, rb = gentest.new_env(max_bt)
    st = s.add_step(0.01)
    def j89asdf():
        st.pulse('rfsoc/dds0/1/freq', CustomSegment((-0.1,)))
    j89asdf()
    comp.finalize()
    with pytest.raises(ValueError, match="Segment time cannot be negative") as exc:
        comp.runtime_finalize(1)
    check_bt(exc, max_bt, 'j89asdf')

    s, comp, rb = gentest.new_env(max_bt)
    st = s.add_step(0.01)
    def ajs89dfjasvsrsdsfa():
        st.pulse('rfsoc/dds0/1/freq', CustomSegment((0,)))
    ajs89dfjasvsrsdsfa()
    comp.finalize()
    with pytest.raises(ValueError,
                       match="Segment time point must monotonically increase") as exc:
        comp.runtime_finalize(1)
    check_bt(exc, max_bt, 'ajs89dfjasvsrsdsfa')

    s, comp, rb = gentest.new_env(max_bt)
    st = s.add_step(0.01)
    def ajs89dfjasvsrsdsf2():
        st.pulse('rfsoc/dds0/1/freq', CustomSegment((0.005, 0.002)))
    ajs89dfjasvsrsdsf2()
    comp.finalize()
    with pytest.raises(ValueError,
                       match="Segment time point must monotonically increase") as exc:
        comp.runtime_finalize(1)
    check_bt(exc, max_bt, 'ajs89dfjasvsrsdsf2')

    s, comp, rb = gentest.new_env(max_bt)
    st = s.add_step(0.01)
    def jas8faslj34ajsdfa8s9():
        st.pulse('rfsoc/dds0/1/freq', CustomSegment((0.005, 0.01)))
    jas8faslj34ajsdfa8s9()
    comp.finalize()
    with pytest.raises(ValueError,
                       match="Segment time point must not exceed action length") as exc:
        comp.runtime_finalize(1)
    check_bt(exc, max_bt, 'jas8faslj34ajsdfa8s9')

    s, comp, rb = gentest.new_env(max_bt)
    st = s.add_step(0.01)
    def jaksdjf8a9sdfjas():
        st.pulse('rfsoc/dds0/1/freq', CustomSegment(([],)))
    jaksdjf8a9sdfjas()
    comp.finalize()
    with pytest.raises(TypeError) as exc:
        comp.runtime_finalize(1)
    check_bt(exc, max_bt, 'jaksdjf8a9sdfjas')

@with_rfsoc_params
def test_val_error(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    # This causes a error to be thrown when converting to boolean
    def j893ajjaks988394():
        s.set('rfsoc/dds0/0/ff', np.array([1, 2]))
    j893ajjaks988394()
    with pytest.raises(ValueError) as exc:
        comp.finalize()
    check_bt(exc, max_bt, 'j893ajjaks988394')

    s, comp, rb = gentest.new_env(max_bt)
    # This causes a error to be thrown when converting to float
    def a88f2398fasd():
        s.set('rfsoc/dds3/0/freq', [1, 2])
    a88f2398fasd()
    with pytest.raises(TypeError) as exc:
        comp.finalize()
    check_bt(exc, max_bt, 'a88f2398fasd')

    s, comp, rb = gentest.new_env(max_bt)
    s.set('rfsoc/dds0/0/ff', True)
    s.add_step(0.01) \
      .pulse('rfsoc/dds1/1/freq', rtval.new_extern(lambda: 1.23)) \
      .pulse('rfsoc/dds2/0/ff', rtval.new_extern(lambda: True))
    # This causes a error to be thrown when converting to boolean
    def js89j308joro82qwe():
        s.set('rfsoc/dds0/0/ff', rtval.new_extern(lambda: np.array([1, 2])))
    js89j308joro82qwe()
    comp.finalize()
    with pytest.raises(TypeError) as exc:
        comp.runtime_finalize(1)
    check_bt(exc, max_bt, 'js89j308joro82qwe')

    s, comp, rb = gentest.new_env(max_bt)
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

    s, comp, rb = gentest.new_env(max_bt)
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

    s, comp, rb = gentest.new_env(max_bt)
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
    s, comp, rb = gentest.new_env(max_bt)

    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/ff', False)
    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/ff', True)
    s.add_step(1e-3) \
      .pulse('rfsoc/dds0/0/ff', True)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(409600), Pulse(819208, ff=True)],
          'amp': [Pulse(409600), Pulse(819208)],
          'phase': [Pulse(409600), Pulse(819208)]
        },
        1: {
          'freq': [Pulse(1228808)],
          'amp': [Pulse(1228808)],
          'phase': [Pulse(1228808)]
        },
    })

    s, comp, rb = gentest.new_env(max_bt)

    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/ff', False)
    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/ff', True)
    s.add_step(1e-3) \
      .pulse('rfsoc/dds0/0/ff', True, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(409600), Pulse(409600, ff=True),
                   Pulse(409608, sync=True, ff=True)],
          'amp': [Pulse(409600), Pulse(409600), Pulse(409608)],
          'phase': [Pulse(409600), Pulse(409600), Pulse(409608)]
        },
        1: {
          'freq': [Pulse(1228808)],
          'amp': [Pulse(1228808)],
          'phase': [Pulse(1228808)]
        },
    })

@with_rfsoc_params
def test_param_output(max_bt):
    s, comp, rb = gentest.new_env(max_bt)

    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/amp', 0)
    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/amp', 0.2)
    s.add_step(1e-3) \
      .pulse('rfsoc/dds0/0/amp', 0.2)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(409600), Pulse(819208)],
          'amp': [Pulse(409600), Pulse(819208, Spline(0.2))],
          'phase': [Pulse(409600), Pulse(819208)]
        },
        1: {
          'freq': [Pulse(1228808)],
          'amp': [Pulse(1228808)],
          'phase': [Pulse(1228808)]
        },
    })

    s, comp, rb = gentest.new_env(max_bt)

    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/amp', 0)
    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/amp', 0.2)
    s.add_step(1e-3) \
      .pulse('rfsoc/dds0/0/amp', 0.2, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(409600), Pulse(409600), Pulse(409608, sync=True)],
          'amp': [Pulse(409600), Pulse(409600, Spline(0.2)),
                  Pulse(409608, Spline(0.2))],
          'phase': [Pulse(409600), Pulse(409600), Pulse(409608)]
        },
        1: {
          'freq': [Pulse(1228808)],
          'amp': [Pulse(1228808)],
          'phase': [Pulse(1228808)]
        },
    })

@with_rfsoc_params
def test_sync_merge(max_bt):
    s, comp, rb = gentest.new_env(max_bt)

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(0)
    s.set('rfsoc/dds0/0/freq', 120e6, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(8, Spline(120e6), sync=True)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
        1: {
          'freq': [Pulse(8)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
    })

    s, comp, rb = gentest.new_env(max_bt)

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(100e-12)
    s.set('rfsoc/dds0/0/freq', 120e6, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(8, Spline(120e6), sync=True)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
        1: {
          'freq': [Pulse(8)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
    })

    s, comp, rb = gentest.new_env(max_bt)

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(2e-9)
    s.set('rfsoc/dds0/0/freq', 120e6, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(4, Spline(120e6), sync=True), Pulse(5, Spline(120e6))],
          'amp': [Pulse(4), Pulse(5)],
          'phase': [Pulse(4), Pulse(5)]
        },
        1: {
          'freq': [Pulse(9)],
          'amp': [Pulse(9)],
          'phase': [Pulse(9)]
        },
    })

    s, comp, rb = gentest.new_env(max_bt)

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(0)
    s.set('rfsoc/dds0/0/freq', 120e6)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(4, Spline(100e6, 110e6, -180e6, 90e6), sync=True),
                   Pulse(4, Spline(120e6))],
          'amp': [Pulse(4), Pulse(4)],
          'phase': [Pulse(4), Pulse(4)]
        },
        1: {
          'freq': [Pulse(8)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
    })

    s, comp, rb = gentest.new_env(max_bt)

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(100e-12)
    s.set('rfsoc/dds0/0/freq', 120e6)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(4, Spline(100e6, 110e6, -180e6, 90e6), sync=True),
                   Pulse(4, Spline(120e6))],
          'amp': [Pulse(4), Pulse(4)],
          'phase': [Pulse(4), Pulse(4)]
        },
        1: {
          'freq': [Pulse(8)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
    })

    s, comp, rb = gentest.new_env(max_bt)

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(2e-9)
    s.set('rfsoc/dds0/0/freq', 120e6)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(4, Spline(100e6, 110e6, -180e6, 90e6), sync=True),
                   Pulse(5, Spline(120e6))],
          'amp': [Pulse(4), Pulse(5)],
          'phase': [Pulse(4), Pulse(5)]
        },
        1: {
          'freq': [Pulse(9)],
          'amp': [Pulse(9)],
          'phase': [Pulse(9)]
        },
    })

@with_rfsoc_params
def test_dyn_seq1(max_bt):
    b1 = True
    v1 = 0.001

    s, comp, rb = gentest.new_env(max_bt)
    s.add_step(0.0005) \
      .pulse('rfsoc/dds0/0/amp', 0.2)
    s.conditional(rtval.new_extern(lambda: b1)) \
      .add_step(rtval.new_extern(lambda: v1)) \
      .set('rfsoc/dds0/0/amp', 0.1)
    s.set('rfsoc/dds0/0/amp', 0.5)
    s.set('rfsoc/dds0/1/amp', 0.1)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(204800), Pulse(409600), Pulse(8)],
          'amp': [Pulse(204800, Spline(0.2)),
                  Pulse(409600, Spline(0.1)),
                  Pulse(8, Spline(0.5))],
          'phase': [Pulse(204800), Pulse(409600), Pulse(8)]
        },
        1: {
          'freq': [Pulse(614400), Pulse(8)],
          'amp': [Pulse(614400), Pulse(8, Spline(0.1))],
          'phase': [Pulse(614400), Pulse(8)]
        },
    })

    b1 = False
    v1 = 0.001
    comp.runtime_finalize(2)
    gentest.check_output({
        0: {
          'freq': [Pulse(204800), Pulse(8)],
          'amp': [Pulse(204800, Spline(0.2)), Pulse(8, Spline(0.5))],
          'phase': [Pulse(204800), Pulse(8)]
        },
        1: {
          'freq': [Pulse(204800), Pulse(8)],
          'amp': [Pulse(204800), Pulse(8, Spline(0.1))],
          'phase': [Pulse(204800), Pulse(8)]
        },
    })

    b1 = True
    v1 = 0
    comp.runtime_finalize(3)
    gentest.check_output({
        0: {
          'freq': [Pulse(204800), Pulse(8)],
          'amp': [Pulse(204800, Spline(0.2)), Pulse(8, Spline(0.5))],
          'phase': [Pulse(204800), Pulse(8)]
        },
        1: {
          'freq': [Pulse(204800), Pulse(8)],
          'amp': [Pulse(204800), Pulse(8, Spline(0.1))],
          'phase': [Pulse(204800), Pulse(8)]
        },
    })

    b1 = False
    v1 = 0
    comp.runtime_finalize(4)
    gentest.check_output({
        0: {
          'freq': [Pulse(204800), Pulse(8)],
          'amp': [Pulse(204800, Spline(0.2)), Pulse(8, Spline(0.5))],
          'phase': [Pulse(204800), Pulse(8)]
        },
        1: {
          'freq': [Pulse(204800), Pulse(8)],
          'amp': [Pulse(204800), Pulse(8, Spline(0.1))],
          'phase': [Pulse(204800), Pulse(8)]
        },
    })

@with_rfsoc_params
def test_tight_output1(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
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
    gentest.check_output({
        0: {
          'freq': [Pulse(12)],
          'amp': [Pulse(12)],
          'phase': [Pulse(12)]
        },
        1: {
          'freq': [Pulse(4, Spline(100e6, 110e6, -180e6, 90e6)),
                   Pulse(8, Spline(200e6))],
          'amp': [Pulse(4, Spline(0.1, 0.55, -0.9, 0.45)), Pulse(8, Spline(0.9))],
          'phase': [Pulse(4, Spline(0.2)), Pulse(8, Spline(0.1))]
        },
        4: {
          'freq': [Pulse(4, ff=True), Pulse(8, ff=True)],
          'amp': [Pulse(4, pytest.approx(Spline(0.3, -0.8, 3, -1.8),
                                         abs=1.5e-5, rel=2e-5)),
                  Pulse(8, Spline(0.3))],
          'phase': [Pulse(4), Pulse(8)]
        },
        5: {
          'freq': [Pulse(12)],
          'amp': [Pulse(12)],
          'phase': [Pulse(12)]
        },
    })

@with_rfsoc_params
def test_tight_output2(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
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
    gentest.check_output({
        0: {
          'freq': [Pulse(12)],
          'amp': [Pulse(12)],
          'phase': [Pulse(12)]
        },
        1: {
          'freq': [Pulse(4, Spline(200e6, -320e6, 600e6, -360e6), sync=True),
                   Pulse(8, Spline(200e6))],
          'amp': [Pulse(4, Spline(0.1, 0.55, -0.9, 0.45)), Pulse(8, Spline(0.9))],
          'phase': [Pulse(4, Spline(0.2)), Pulse(8, Spline(0.1))]
        },
        4: {
          'freq': [Pulse(4, ff=True), Pulse(8, ff=True)],
          'amp': [Pulse(4, pytest.approx(Spline(0.3, -0.8, 3, -1.8),
                                         abs=1.5e-5, rel=2e-5)),
                  Pulse(8, Spline(0.3))],
          'phase': [Pulse(4), Pulse(8)]
        },
        5: {
          'freq': [Pulse(12)],
          'amp': [Pulse(12)],
          'phase': [Pulse(12)]
        },
    })

@with_rfsoc_params
def test_tight_output3(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
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
    gentest.check_output({
        0: {
          'freq': [Pulse(12)],
          'amp': [Pulse(12)],
          'phase': [Pulse(12)]
        },
        1: {
          'freq': [Pulse(4, Spline(120e6), sync=True), Pulse(8, Spline(200e6))],
          'amp': [Pulse(4, Spline(0.1, 0.55, -0.9, 0.45)), Pulse(8, Spline(0.9))],
          'phase': [Pulse(4, Spline(0.2)), Pulse(8, Spline(0.1))]
        },
        4: {
          'freq': [Pulse(4, ff=True), Pulse(8, ff=True)],
          'amp': [Pulse(4, pytest.approx(Spline(0.3, -0.8, 3, -1.8),
                                         abs=1.5e-5, rel=2e-5)),
                  Pulse(8, Spline(0.3))],
          'phase': [Pulse(4), Pulse(8)]
        },
        5: {
          'freq': [Pulse(12)],
          'amp': [Pulse(12)],
          'phase': [Pulse(12)]
        },
    })

@with_rfsoc_params
def test_tight_output4(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
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
    gentest.check_output({
        0: {
          'freq': [Pulse(12)],
          'amp': [Pulse(12)],
          'phase': [Pulse(12)]
        },
        1: {
          'freq': [Pulse(4, Spline(200e6, -320e6, 600e6, -360e6), sync=True),
                   Pulse(8, Spline(200e6))],
          'amp': [Pulse(4, Spline(0.1, 0.55, -0.9, 0.45)), Pulse(8, Spline(0.9))],
          'phase': [Pulse(4, Spline(0.2)), Pulse(8, Spline(0.1))]
        },
        4: {
          'freq': [Pulse(4, ff=True), Pulse(8, ff=True)],
          'amp': [Pulse(4, pytest.approx(Spline(0.3, -0.8, 3, -1.8),
                                         abs=1.5e-5, rel=2e-5)),
                  Pulse(8, Spline(0.3))],
          'phase': [Pulse(4), Pulse(8)]
        },
        5: {
          'freq': [Pulse(12)],
          'amp': [Pulse(12)],
          'phase': [Pulse(12)]
        },
    })

@with_rfsoc_params
def test_tight_output5(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
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
    gentest.check_output({
        0: {
          'freq': [Pulse(12)],
          'amp': [Pulse(12)],
          'phase': [Pulse(12)]
        },
        1: {
          'freq': [Pulse(4, Spline(200e6, -320e6, 600e6, -360e6), sync=True),
                   Pulse(8, Spline(200e6))],
          'amp': [Pulse(4, Spline(0.1, 0.55, -0.9, 0.45)), Pulse(8, Spline(0.9))],
          'phase': [Pulse(4, Spline(0.2)), Pulse(8, Spline(0.1))]
        },
        4: {
          'freq': [Pulse(4, ff=True), Pulse(8, ff=True)],
          'amp': [Pulse(4, pytest.approx(Spline(0.3, -0.8, 3, -1.8),
                                         abs=1.5e-5, rel=2e-5)),
                  Pulse(8, Spline(0.3))],
          'phase': [Pulse(4),
                    Pulse(8)]
        },
        5: {
          'freq': [Pulse(12)],
          'amp': [Pulse(12)],
          'phase': [Pulse(12)]
        },
    })

@with_rfsoc_params
def test_dds_delay_rt_error(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    rb.set_dds_delay(0, rtval.new_extern(lambda: -0.001))
    comp.finalize()
    with pytest.raises(ValueError, match="DDS time offset -0.001 cannot be negative."):
        comp.runtime_finalize(1)

    s, comp, rb = gentest.new_env(max_bt)
    rb.set_dds_delay(1, rtval.new_extern(lambda: 1))
    comp.finalize()
    with pytest.raises(ValueError,
                       match="DDS time offset 1.0 cannot be more than 100ms."):
        comp.runtime_finalize(1)

@with_rfsoc_params
@pytest.mark.parametrize('use_rt', [False, True])
def test_dds_delay(max_bt, use_rt):
    def wrap_value(v):
        if use_rt:
            return rtval.new_extern(lambda: v)
        return v
    s, comp, rb = gentest.new_env(max_bt)
    with pytest.raises(ValueError, match="DDS time offset -0.001 cannot be negative."):
        rb.set_dds_delay(0, -0.001)
    with pytest.raises(ValueError,
                       match="DDS time offset 1.0 cannot be more than 100ms."):
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
    gentest.check_output({
        0: {
          'freq': [Pulse(410), Pulse(409600), Pulse(409198)],
          'amp': [Pulse(410), Pulse(409600, Spline(0.1)), Pulse(409198)],
          'phase': [Pulse(410), Pulse(409600), Pulse(409198)]
        },
        1: {
          'freq': [Pulse(410), Pulse(818798)],
          'amp': [Pulse(410), Pulse(818798)],
          'phase': [Pulse(410), Pulse(818798, Spline(0.2))]
        },
        2: {
          'freq': [Pulse(819208)],
          'amp': [Pulse(819208)],
          'phase': [Pulse(819208)]
        },
        3: {
          'freq': [Pulse(409600), Pulse(409608, Spline(100e6))],
          'amp': [Pulse(409600), Pulse(409608)],
          'phase': [Pulse(409600), Pulse(409608)]
        },
        4: {
          'freq': [Pulse(409600), Pulse(409608)],
          'amp': [Pulse(409600, Spline(0.3)), Pulse(409608)],
          'phase': [Pulse(409600), Pulse(409608)]
        },
        5: {
          'freq': [Pulse(819208)],
          'amp': [Pulse(819208)],
          'phase': [Pulse(819208)]
        },
    })

@with_rfsoc_params
def test_cond_ramp_error(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    s.conditional(rtval.new_extern(lambda: False)) \
      .add_step(rtval.new_extern(lambda: 0)) \
      .set('rfsoc/dds0/0/amp', DivLengthFunction()) \
      .pulse('rfsoc/dds0/1/amp', DivLengthFunction())
    comp.finalize()
    comp.runtime_finalize(1)

    s, comp, rb = gentest.new_env(max_bt)
    s.add_step(0) \
      .set('rfsoc/dds0/0/amp', Blackman(1)) \
      .pulse('rfsoc/dds0/1/amp', Blackman(1)) \
      .set('rfsoc/dds1/0/amp', BlackmanSquare(1)) \
      .pulse('rfsoc/dds1/1/amp', BlackmanSquare(1))
    comp.finalize()
    comp.runtime_finalize(1)

    s, comp, rb = gentest.new_env(max_bt)
    s.add_step(rtval.new_extern(lambda: 0)) \
      .set('rfsoc/dds0/0/amp', Blackman(1)) \
      .pulse('rfsoc/dds0/1/amp', Blackman(1)) \
      .set('rfsoc/dds1/0/amp', BlackmanSquare(1)) \
      .pulse('rfsoc/dds1/1/amp', BlackmanSquare(1))
    comp.finalize()
    comp.runtime_finalize(1)

@with_rfsoc_params
def test_cross_channel_sync1(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    @s.add_background
    def other_step(s):
        s.wait(2.5e-3)
        s.set('rfsoc/dds0/1/amp', 0.0)
        s.wait(2.5e-3)
        s.set('rfsoc/dds0/1/phase', 0.0)
        s.wait(2.5e-3)
        s.set('rfsoc/dds0/1/ff', False)
    s.add_step(10e-3) \
      .set('rfsoc/dds0/1/freq', LinearRamp(100e6, 120e6))
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(4096008)],
          'amp': [Pulse(4096008)],
          'phase': [Pulse(4096008)]
        },
        1: {
          'freq': [Pulse(4096000, Spline(100e6, 20e6)), Pulse(8, Spline(120e6))],
          'amp': [Pulse(4096000), Pulse(8)],
          'phase': [Pulse(4096000), Pulse(8)]
        },
    })

    s, comp, rb = gentest.new_env(max_bt)
    @s.add_background
    def other_step(s):
        s.wait(2.5e-3)
        s.set('rfsoc/dds0/1/amp', 0.0, sync=True)
        s.wait(2.5e-3)
        s.set('rfsoc/dds0/1/phase', 0.0, sync=True)
        s.wait(2.5e-3)
        s.set('rfsoc/dds0/1/ff', False, sync=True)
    s.add_step(10e-3) \
      .set('rfsoc/dds0/1/freq', LinearRamp(100e6, 120e6))
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(4096008)],
          'amp': [Pulse(4096008)],
          'phase': [Pulse(4096008)]
        },
        1: {
          'freq': [Pulse(1024000, Spline(100e6, 5e6)),
                   Pulse(1024000, Spline(105e6, 5e6), sync=True),
                   Pulse(1024000, Spline(110e6, 5e6), sync=True),
                   Pulse(1024000, Spline(115e6, 5e6), sync=True),
                   Pulse(8, Spline(120e6))],
          'amp': [Pulse(1024000), Pulse(1024000), Pulse(1024000),
                  Pulse(1024000), Pulse(8)],
          'phase': [Pulse(1024000), Pulse(1024000), Pulse(1024000),
                    Pulse(1024000), Pulse(8)]
        },
    })

@with_rfsoc_params
def test_cross_channel_sync2(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    @s.add_background
    def amp_step(s):
        s.wait(5e-6)
        s.set('rfsoc/dds0/1/amp', 0.0, sync=True)
    @s.add_background
    def freq_step(s):
        s.wait(5e-6)
        s.set('rfsoc/dds0/1/freq', 100e6)
        s.add_step(5e-6) \
          .set('rfsoc/dds0/1/freq', 110e6)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(4104)],
          'amp': [Pulse(4104)],
          'phase': [Pulse(4104)]
        },
        1: {
          'freq': [Pulse(2048), Pulse(2056, Spline(110e6), sync=True)],
          'amp': [Pulse(2048), Pulse(2056)],
          'phase': [Pulse(2048), Pulse(2056)]
        },
    })

    s, comp, rb = gentest.new_env(max_bt)
    @s.add_step
    def freq_step(s):
        s.wait(5e-6)
        s.set('rfsoc/dds0/1/freq', 100e6)
        s.add_step(5e-6) \
          .set('rfsoc/dds0/1/phase', 0.0, sync=True) \
          .set('rfsoc/dds0/1/freq', 110e6)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(4104)],
          'amp': [Pulse(4104)],
          'phase': [Pulse(4104)]
        },
        1: {
          'freq': [Pulse(2048), Pulse(2056, Spline(110e6), sync=True)],
          'amp': [Pulse(2048), Pulse(2056)],
          'phase': [Pulse(2048), Pulse(2056)]
        },
    })

    s, comp, rb = gentest.new_env(max_bt)
    @s.add_step
    def freq_step(s):
        s.wait(5e-6)
        s.set('rfsoc/dds0/1/freq', 100e6)
        s.set('rfsoc/dds0/1/ff', False, sync=True)
        s.add_step(5e-6) \
          .set('rfsoc/dds0/1/freq', 110e6)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(4104)],
          'amp': [Pulse(4104)],
          'phase': [Pulse(4104)]
        },
        1: {
          'freq': [Pulse(2048),
                   Pulse(4, Spline(100e6, 55e6, -90e6, 45e6), sync=True),
                   Pulse(2052, Spline(110e6))],
          'amp': [Pulse(2048), Pulse(4), Pulse(2052)],
          'phase': [Pulse(2048), Pulse(4), Pulse(2052)]
        },
    })

@with_rfsoc_params
def test_use_all_channels(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    assert not rb.use_all_channels
    rb.use_all_channels = True
    s.add_step(5e-6) \
      .set('rfsoc/dds0/1/freq', 100e6) \
      .set('rfsoc/dds0/1/amp', 0.2)
    comp.finalize()

    comp.runtime_finalize(1)
    unused = {
        'freq': [Pulse(2056)],
        'amp': [Pulse(2056)],
        'phase': [Pulse(2056)]
    }
    gentest.check_output({
        0: unused,
        1: {
          'freq': [Pulse(2056, Spline(100e6))],
          'amp': [Pulse(2056, Spline(0.2))],
          'phase': [Pulse(2056)]
        },
        2: unused,
        3: unused,
        4: unused,
        5: unused,
        6: unused,
        7: unused,
        8: unused,
        9: unused,
        10: unused,
        11: unused,
        12: unused,
        13: unused,
        14: unused,
        15: unused,
        16: unused,
        17: unused,
        18: unused,
        19: unused,
        20: unused,
        21: unused,
        22: unused,
        23: unused,
        24: unused,
        25: unused,
        26: unused,
        27: unused,
        28: unused,
        29: unused,
        30: unused,
        31: unused,
        32: unused,
        33: unused,
        34: unused,
        35: unused,
        36: unused,
        37: unused,
        38: unused,
        39: unused,
        40: unused,
        41: unused,
        42: unused,
        43: unused,
        44: unused,
        45: unused,
        46: unused,
        47: unused,
        48: unused,
        49: unused,
        50: unused,
        51: unused,
        52: unused,
        53: unused,
        54: unused,
        55: unused,
        56: unused,
        57: unused,
        58: unused,
        59: unused,
        60: unused,
        61: unused,
        62: unused,
        63: unused,
    })
    s, comp, rb = gentest.new_env(max_bt)
    assert not rb.use_all_channels
    rb.use_all_channels = True
    s.wait(5e-6)
    comp.finalize()

    comp.runtime_finalize(1)
    unused = {
        'freq': [Pulse(2056)],
        'amp': [Pulse(2056)],
        'phase': [Pulse(2056)]
    }
    gentest.check_output({})

@with_rfsoc_params
def test_long_wait(max_bt):
    s, comp, rb = gentest.new_env(max_bt)
    s.add_step(5000) \
      .set('rfsoc/dds0/1/freq', 100e6) \
      .set('rfsoc/dds0/1/amp', 0.2)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(1024000000004), Pulse(1024000000004)],
          'amp': [Pulse(1024000000004), Pulse(1024000000004)],
          'phase': [Pulse(1024000000004), Pulse(1024000000004)]
        },
        1: {
          'freq': [Pulse(1024000000004, Spline(100e6)),
                   Pulse(1024000000004, Spline(100e6))],
          'amp': [Pulse(1024000000004, Spline(0.2)),
                  Pulse(1024000000004, Spline(0.2))],
          'phase': [Pulse(1024000000004), Pulse(1024000000004)]
        },
    })

    s, comp, rb = gentest.new_env(max_bt)
    s.add_step(5000) \
      .set('rfsoc/dds0/1/freq', LinearRamp(50e6, 100e6), sync=True) \
      .set('rfsoc/dds0/1/amp', 0.2)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(1024000000004), Pulse(1024000000004)],
          'amp': [Pulse(1024000000004), Pulse(1024000000004)],
          'phase': [Pulse(1024000000004), Pulse(1024000000004)]
        },
        1: {
          'freq': [Pulse(1024000000000, pytest.approx(Spline(50e6, 25e6), abs=0.1),
                         sync=True),
                   Pulse(1024000000000, pytest.approx(Spline(75e6, 25e6), abs=0.1)),
                   Pulse(8, Spline(100e6))],
          'amp': [Pulse(1024000000000, Spline(0.2)),
                  Pulse(1024000000000, Spline(0.2)),
                  Pulse(8, Spline(0.2))],
          'phase': [Pulse(1024000000000), Pulse(1024000000000), Pulse(8)]
        },
    })

    s, comp, rb = gentest.new_env(max_bt)
    s.add_step(10000) \
      .set('rfsoc/dds0/1/freq', 100e6) \
      .set('rfsoc/dds0/1/amp', 0.2)
    comp.finalize()

    comp.runtime_finalize(1)
    gentest.check_output({
        0: {
          'freq': [Pulse(1099511627775), Pulse(1099511627775),
                   Pulse(948488372229), Pulse(948488372229)],
          'amp': [Pulse(1099511627775), Pulse(1099511627775),
                  Pulse(948488372229), Pulse(948488372229)],
          'phase': [Pulse(1099511627775), Pulse(1099511627775),
                    Pulse(948488372229), Pulse(948488372229)]
        },
        1: {
          'freq': [Pulse(1099511627775, Spline(100e6)),
                   Pulse(1099511627775, Spline(100e6)),
                   Pulse(948488372229, Spline(100e6)),
                   Pulse(948488372229, Spline(100e6))],
          'amp': [Pulse(1099511627775, Spline(0.2)),
                  Pulse(1099511627775, Spline(0.2)),
                  Pulse(948488372229, Spline(0.2)),
                  Pulse(948488372229, Spline(0.2))],
          'phase': [Pulse(1099511627775), Pulse(1099511627775),
                    Pulse(948488372229), Pulse(948488372229)]
        },
    })
