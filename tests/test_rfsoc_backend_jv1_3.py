#

from brassboard_seq.action import Blackman, BlackmanSquare, LinearRamp, SeqCubicSpline
from brassboard_seq import rfsoc_backend, rtval
import py_test_utils as test_utils
from rfsoc_test_utils import Env as _Env, Spline, Pulse, approx_pulses

import pytest
import numpy as np

class ParamPulses:
    def __init__(self):
        self.cycles = 0
        self.pulses = []

    def add(self, d):
        cycles = d['cycles']
        assert (self.cycles == 0) == d['trig']
        modtype = d['modtype'][0]
        if modtype in ('frame_rot0', 'frame_rot1'):
            assert d['fwd'] == 0
            assert d['inv'] == 0
            assert not d['eof']
            assert not d['clr']
            assert d['spline'] == [0.0, 0.0, 0.0, 0.0]
        else:
            assert not d['enable']
            if not modtype.startswith('freq'):
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
        modtype = d['modtype'][0]
        if modtype.startswith('freq'):
            self.freq.add(d)
        elif modtype.startswith('amp'):
            self.amp.add(d)
        elif modtype.startswith('phase'):
            self.phase.add(d)
        elif modtype.startswith('frame_rot'):
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

class Env(_Env):
    def __init__(self):
        super().__init__(rfsoc_backend.Jaqalv1_3Generator())

    def get_output(self):
        tones = {
            'freq0': 0, 'amp0': 0, 'phase0': 0, 'frame_rot0': 0,
            'freq1': 1, 'amp1': 1, 'phase1': 1, 'frame_rot1': 1,
        }
        pulses = [TonePulses() for _ in range(64)]
        for board_id in range(4):
            prefix = self.gen.get_prefix(board_id)
            sequence = self.gen.get_sequence(board_id)
            for inst in rfsoc_backend.Jaqal_v1_3.extract_pulses(prefix + sequence):
                d = inst.to_dict()
                assert d['type'] == 'pulse_data'
                chn = d['channel']
                modtypes = d['modtype']
                assert len(modtypes) == 1
                modtype = modtypes[0]
                tone = tones[modtype]
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

test_env = Env()

@test_utils.with_seq_params
def test_channels(max_bt):
    comp = test_env.new_comp(max_bt)
    assert not comp.rb.has_output
    comp.finalize()
    assert not comp.rb.has_output

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    assert s.get_channel_id('artiq/ttl0') == 0
    assert s.get_channel_id('rfsoc/dds1/0/amp') == 1
    assert s.get_channel_id('rfsoc/dds1/1/freq') == 2
    assert s.get_channel_id('artiq/ttl2') == 3
    assert s.get_channel_id('rfsoc/dds2/0/phase') == 4
    assert s.get_channel_id('rfsoc/dds0/0/ff') == 5
    assert s.get_channel_id('rfsoc/dds1/1/amp') == 6
    assert s.get_channel_id('rfsoc/dds1/1/ff') == 7
    assert s.get_channel_id('rfsoc/dds0/0/freq') == 8
    assert not comp.rb.has_output
    comp.finalize()
    assert comp.rb.has_output
    channels = comp.get_channel_info()
    chn_ids = [tone_chn.chn for tone_chn in channels.channels]
    assert chn_ids == [2, 3, 4, 0, 1, 5]
    assert channels.chn_map == {1: (0, 'amp'), 2: (1, 'freq'),
                                4: (2, 'phase'), 5: (3, 'ff'),
                                6: (1, 'amp'), 7: (1, 'ff'),
                                8: (3, 'freq')}
    comp.runtime_finalize(1)
    with pytest.raises(IndexError, match="Board index should be in \\[0, 3\\]"):
        test_env.gen.get_prefix(-1)
    with pytest.raises(IndexError, match="Board index should be in \\[0, 3\\]"):
        test_env.gen.get_prefix(4)
    with pytest.raises(IndexError, match="Board index should be in \\[0, 3\\]"):
        test_env.gen.get_sequence(-1)
    with pytest.raises(IndexError, match="Board index should be in \\[0, 3\\]"):
        test_env.gen.get_sequence(4)
    test_env.check_output({
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

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    assert s.get_channel_id('rfsoc/dds0/0') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds0/0'):
        comp.finalize()

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    assert s.get_channel_id('rfsoc/dds0/0/freq/a') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds0/0/freq/a'):
        comp.finalize()

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    assert s.get_channel_id('rfsoc/ch0/0/freq') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/ch0/0/freq'):
        comp.finalize()

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    assert s.get_channel_id('rfsoc/dds50/0/freq') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds50/0/freq'):
        comp.finalize()

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    assert s.get_channel_id('rfsoc/dds10/2/freq') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds10/2/freq'):
        comp.finalize()

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    assert s.get_channel_id('rfsoc/dds10/0/param') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds10/0/param'):
        comp.finalize()

@test_utils.with_seq_params
def test_output1(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.add_step(0.01) \
      .pulse('rfsoc/dds0/1/amp', 0.2) \
      .set('rfsoc/dds0/1/freq', 100e6) \
      .set('rfsoc/dds0/1/phase', 0.1) \
      .set('rfsoc/dds0/1/ff', True)
    comp.finalize()
    channels = comp.get_channel_info()
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
    test_env.check_output({
        0: {
          'freq': [Pulse(4096008)],
          'amp': [Pulse(4096008)],
          'phase': [Pulse(4096008)]
        },
        1: {
          'freq': [Pulse(4096008, Spline(100e6), ff=True)],
          'amp': [Pulse(4096000, Spline(0.2)), Pulse(8)],
          'phase': [Pulse(4096008, Spline(0.1))]
        },
    })

@test_utils.with_seq_params
def test_output2(max_bt):
    b1 = True
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.conditional(rtval.new_extern(lambda: b1)) \
      .add_step(rtval.new_extern(lambda: 0.01)) \
      .pulse('rfsoc/dds0/1/amp', rtval.new_extern(lambda: 0.2)) \
      .set('rfsoc/dds0/1/freq', rtval.new_extern(lambda: 100e6), sync=True) \
      .set('rfsoc/dds0/1/phase', rtval.new_extern(lambda: 0.1)) \
      .set('rfsoc/dds0/1/ff', rtval.new_extern(lambda: True), sync=False)
    comp.finalize()
    channels = comp.get_channel_info()
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
    test_env.check_output({
        0: {
          'freq': [Pulse(4096008)],
          'amp': [Pulse(4096008)],
          'phase': [Pulse(4096008)]
        },
        1: {
          'freq': [Pulse(4096008, Spline(100e6), sync=True, ff=True)],
          'amp': [Pulse(4096000, Spline(0.2)), Pulse(8)],
          'phase': [Pulse(4096008, Spline(0.1))]
        },
    })

    b1 = False
    comp.runtime_finalize(2)
    test_env.check_output({
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

@test_utils.with_seq_params
def test_output3(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.conditional(False) \
      .add_step(rtval.new_extern(lambda: 0.01)) \
      .pulse('rfsoc/dds0/1/amp', rtval.new_extern(lambda: 0.2)) \
      .set('rfsoc/dds0/1/freq', rtval.new_extern(lambda: 100e6), sync=True) \
      .set('rfsoc/dds0/1/phase', rtval.new_extern(lambda: 0.1)) \
      .set('rfsoc/dds0/1/ff', rtval.new_extern(lambda: True), sync=False)
    comp.finalize()
    channels = comp.get_channel_info()
    assert len(channels.channels) == 2
    assert channels.channels[0].chn == 1
    assert channels.channels[1].chn == 0
    assert len(channels.channels[0].actions[0]) == 0 # freq
    assert len(channels.channels[0].actions[1]) == 0 # phase
    assert len(channels.channels[0].actions[2]) == 0 # amp
    assert len(channels.channels[0].actions[3]) == 0 # ff

    comp.runtime_finalize(1)
    test_env.check_output({
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

@test_utils.with_seq_params
def test_ramp_output1(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    ramp1 = test_utils.StaticFunction()
    ramp2 = test_utils.StaticFunction()
    s.add_step(0.01) \
      .pulse('rfsoc/dds0/1/amp', ramp1) \
      .set('rfsoc/dds0/1/phase', ramp2)
    comp.finalize()
    channels = comp.get_channel_info()
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
    test_env.check_output({
        0: {
          'freq': [Pulse(4096008)],
          'amp': [Pulse(4096008)],
          'phase': [Pulse(4096008)]
        },
        1: {
          'freq': [Pulse(4096008)],
          'amp': [Pulse(2048000, Spline(-0.01, 0.0025)),
                  Pulse(2048000, Spline(-0.0075, 0.0025)),
                  Pulse(8)],
          'phase': [Pulse(2048000, Spline(-0.01, 0.0025)),
                    Pulse(2048000, Spline(-0.0075, 0.0025)),
                    Pulse(8, Spline(-0.005))]
        },
    })

@test_utils.with_seq_params
def test_ramp_output2(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    ramp1 = test_utils.StaticFunction()
    ramp2 = test_utils.StaticFunction()
    s.add_step(rtval.new_extern(lambda: 0.01)) \
      .pulse('rfsoc/dds0/1/amp', ramp1) \
      .set('rfsoc/dds0/1/phase', ramp2)
    comp.finalize()
    channels = comp.get_channel_info()
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
    test_env.check_output({
        0: {
          'freq': [Pulse(4096008)],
          'amp': [Pulse(4096008)],
          'phase': [Pulse(4096008)]
        },
        1: {
          'freq': [Pulse(4096008)],
          'amp': [Pulse(2048000, Spline(-0.01, 0.0025)),
                  Pulse(2048000, Spline(-0.0075, 0.0025)),
                  Pulse(8)],
          'phase': [Pulse(2048000, Spline(-0.01, 0.0025)),
                    Pulse(2048000, Spline(-0.0075, 0.0025)),
                    Pulse(8, Spline(-0.005))]
        },
    })

@test_utils.with_seq_params
def test_ramp_output3(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    ramp1 = test_utils.LinearRampNoSeg(0.1, 0.2)
    ramp2 = LinearRamp(0.1, 0.2)
    ramp3 = SeqCubicSpline(0.2, 0.1, 0.9, 0.3)
    s.add_step(5e-3) \
      .pulse('rfsoc/dds0/1/amp', ramp1) \
      .set('rfsoc/dds2/1/phase', ramp2) \
      .set('rfsoc/dds4/1/amp', ramp3) \
      .set('rfsoc/dds2/0/amp', 0.1)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048008)],
          'phase': [Pulse(2048008)]
        },
        1: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(1024000, Spline(0.1, 0.05)),
                  Pulse(1024000, Spline(0.15, 0.05)),
                  Pulse(8)],
          'phase': [Pulse(2048008)]
        },
        4: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048008, Spline(0.1))],
          'phase': [Pulse(2048008)]
        },
        5: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048008)],
          'phase': [Pulse(2048000, Spline(0.1, 0.1)), Pulse(8, Spline(0.2))]
        },
        8: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048008)],
          'phase': [Pulse(2048008)]
        },
        9: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048000, Spline(0.2, 0.1, 0.9, 0.3)),
                  Pulse(8, pytest.approx(Spline(-0.5), abs=4e-5))],
          'phase': [Pulse(2048008)],
        },
    })

@test_utils.with_seq_params
def test_ramp_output4(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    ramp1 = test_utils.LinearRampNoSeg(0.1, 0.2)
    ramp2 = SeqCubicSpline(0.2, 0.1, 0.9, 0.3)
    s.add_step(5e-3) \
      .pulse('rfsoc/dds0/1/phase', ramp1) \
      .set('rfsoc/dds2/1/phase', ramp2) \
      .set('rfsoc/dds2/0/amp', 0.1) \
      .set('rfsoc/dds2/0/phase', -1 / 2 / np.pi)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048008)],
          'phase': [Pulse(2048008)]
        },
        1: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048008)],
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
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048008)],
          'phase': [Pulse(2048000, Spline(0.2, 0.1, 0.9, 0.3)),
                    Pulse(8, Spline(-0.5))]
        },
    })

@test_utils.with_seq_params
def test_ramp_output5(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.set('rfsoc/dds0/1/amp', 0.2)
    s.set('rfsoc/dds2/0/freq', 80e6)
    s.set('rfsoc/dds0/1/phase', 0.9)
    s.add_step(5e-3) \
      .set('rfsoc/dds0/1/amp', test_utils.RampUpAndDown(0.25, 0.5)) \
      .set('rfsoc/dds2/0/freq', test_utils.RampUpAndDown(0.5, 100e6)) \
      .set('rfsoc/dds0/1/phase', test_utils.RampUpAndDown(0.75, 0.3)) \
      .set('rfsoc/dds0/0/amp', 0.2) \
      .set('rfsoc/dds0/0/phase', -0.5 / 2 / np.pi)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048008, Spline(0.2))],
          'phase': [Pulse(2048008, Spline(-0.5 / 2 / np.pi))]
        },
        1: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(512000, Spline(0.2, 0.3)),
                  Pulse(1536000, Spline(0.5, -0.3)),
                  Pulse(8, Spline(0.2))],
          'phase': [Pulse(1536000, Spline(-0.1, -0.6)),
                    Pulse(512000, Spline(0.3, 0.6)),
                    Pulse(8, Spline(-0.1))]
        },
        4: {
          'freq': [Pulse(1024000, Spline(80e6, 20e6)),
                   Pulse(1024000, Spline(100e6, -20e6)),
                   Pulse(8, Spline(80e6))],
          'amp': [Pulse(2048008)],
          'phase': [Pulse(2048008)]
        },
        5: {
          'freq': [Pulse(2048008)],
          'amp': [Pulse(2048008)],
          'phase': [Pulse(2048008)]
        },
    })

@test_utils.with_seq_params
def test_ramp_output6(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
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
    test_env.check_output({
        0: {
          'freq': [
            Pulse(409600),
            # freq 0.0
            Pulse(819200, Spline(100e6, 5e6, -5e6, 2.5e6)),
            # freq 0.1
            Pulse(819200, Spline(102.5e6, 2.5e6, 2.5e6, 2.5e6), ff=True),
            # freq 1.0
            Pulse(819200, Spline(10e6, 50e6, -7.5e6, -2.5e6), ff=True),
            # freq 1.1
            Pulse(819200, Spline(50e6, 27.5e6, -15e6, -2.5e6)),
            Pulse(819208, Spline(60e6))],
          'amp': [
            # amp 0
            Pulse(1638400, Spline(0.2, -0.1, 0.9, -0.3)),
            # amp 1
            Pulse(1638400, Spline(0.5, -0.1, 0.5, -0.2)),
            Pulse(1228808, Spline(0.7))
          ],
          'phase': [
            Pulse(819200),
            # phase 0
            Pulse(1638400, Spline(0.4, -0.2, 0.1, -0.2)),
            # phase 1
            Pulse(1638400, Spline(0.2, 0.7, 0.1, -0.9)),
            Pulse(409608, Spline(0.1))
            ]
        },
        1: {
          'freq': [Pulse(4505608)],
          'amp': [Pulse(4505608)],
          'phase': [Pulse(4505608)]
        },
        4: {
          'freq': [Pulse(4505608)],
          'amp': [Pulse(409600), Pulse(4096008, Spline(0.1))],
          'phase': [Pulse(4505608)]
        },
        5: {
          'freq': [Pulse(4505608)],
          'amp': [Pulse(4505608)],
          'phase': [Pulse(4505608)]
        },
    })

@test_utils.with_seq_params
def test_ramp_output7(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.add_step(1) \
      .set('rfsoc/dds0/0/amp', Blackman(1.0)) \
      .set('rfsoc/dds0/1/amp', 1.0)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(409600008)],
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
          'phase': [Pulse(409600008)]
        },
        1: {
          'freq': [Pulse(409600008)],
          'amp': [Pulse(409600008, Spline(1.0))],
          'phase': [Pulse(409600008)]
        }
    })

@test_utils.with_seq_params
def test_ramp_reuse_spline(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    r = SeqCubicSpline(0.1, 0.4, -0.2, 0.1)
    s.add_step(5e-3) \
      .set('rfsoc/dds0/0/amp', r)
    s.add_step(10e-3) \
      .set('rfsoc/dds0/0/amp', r)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
            'freq': [Pulse(6144008)],
            'amp': [Pulse(2048000, Spline(0.1, 0.4, -0.2, 0.1)),
                    Pulse(4096000, Spline(0.1, 0.4, -0.2, 0.1)),
                    Pulse(8, Spline(0.4))],
            'phase': [Pulse(6144008)],
        },
        1: {
            'freq': [Pulse(6144008)],
            'amp': [Pulse(6144008)],
            'phase': [Pulse(6144008)],
        },
    })

@test_utils.with_seq_params
def test_ramp_reuse_oldval(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    r = test_utils.RampUpAndDown(0.5, 0.9)
    s.add_step(5e-3) \
      .set('rfsoc/dds0/0/amp', r)
    s.set('rfsoc/dds0/0/amp', 0.3)
    s.add_step(10e-3) \
      .set('rfsoc/dds0/0/amp', r)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
            'freq': [Pulse(6144008)],
            'amp': [Pulse(1024000, Spline(0, 0.9)), Pulse(1024000, Spline(0.9, -0.9)),
                    Pulse(2048000, Spline(0.3, 0.6)), Pulse(2048000, Spline(0.9, -0.6)),
                    Pulse(8, Spline(0.3))],
            'phase': [Pulse(6144008)],
        },
        1: {
            'freq': [Pulse(6144008)],
            'amp': [Pulse(6144008)],
            'phase': [Pulse(6144008)],
        },
    })

@test_utils.with_seq_params
def test_ramp_reuse_linear(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    r = LinearRamp(0.1, 0.9)
    r2 = test_utils.LinearRampNoSeg(0.1, 0.9)
    s.add_step(5e-3) \
      .set('rfsoc/dds0/0/amp', r) \
      .set('rfsoc/dds0/1/amp', r2)
    s.add_step(10e-3) \
      .set('rfsoc/dds0/0/amp', r2) \
      .set('rfsoc/dds0/1/amp', r)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
            'freq': [Pulse(6144008)],
            'amp': [Pulse(2048000, Spline(0.1, 0.8)), Pulse(2048000, Spline(0.1, 0.4)),
                    Pulse(2048000, Spline(0.5, 0.4)), Pulse(8, Spline(0.9))],
            'phase': [Pulse(6144008)],
        },
        1: {
            'freq': [Pulse(6144008)],
            'amp': [Pulse(1024000, Spline(0.1, 0.4)), Pulse(1024000, Spline(0.5, 0.4)),
                    Pulse(4096000, Spline(0.1, 0.8)), Pulse(8, Spline(0.9))],
            'phase': [Pulse(6144008)],
        },
    })

@test_utils.with_seq_params
def test_ramp_reuse_blackman(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    r = Blackman(1.0)
    s.add_step(1) \
      .set('rfsoc/dds0/0/amp', r) \
      .set('rfsoc/dds0/1/amp', 1.0)
    s.add_step(2) \
      .set('rfsoc/dds0/0/amp', r) \
      .set('rfsoc/dds0/1/amp', 1.0)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(1228800008)],
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
          'phase': [Pulse(1228800008)]
        },
        1: {
          'freq': [Pulse(1228800008)],
          'amp': [Pulse(1228800008, Spline(1.0))],
          'phase': [Pulse(1228800008)]
        }
    })

@test_utils.with_seq_params
def test_short_ramp_output1(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    ramp1 = test_utils.StaticFunction()
    ramp2 = test_utils.StaticFunction()
    s.add_step(200e-9) \
      .pulse('rfsoc/dds0/1/amp', ramp1) \
      .set('rfsoc/dds0/1/phase', ramp2)
    comp.finalize()
    channels = comp.get_channel_info()
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
    test_env.check_output({
        0: {
          'freq': [Pulse(90)],
          'amp': [Pulse(90)],
          'phase': [Pulse(90)]
        },
        1: {
          'freq': [Pulse(90)],
          'amp': [Pulse(41, Spline(-2e-7, 5e-8)),
                  Pulse(41, Spline(-1.5e-7, 5e-8)),
                  Pulse(8)],
          'phase': [Pulse(41, Spline(-2e-7, 0.5e-7)),
                    Pulse(41, Spline(-1.5e-7, 0.5e-7)),
                    Pulse(8, Spline(-1e-7))]
        },
    })

@test_utils.with_seq_params
def test_short_ramp_output2(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    ramp1 = test_utils.LinearRampNoSeg(0.1, 0.2)
    ramp2 = LinearRamp(0.1, 0.2)
    s.add_step(5e-9) \
      .pulse('rfsoc/dds0/1/amp', ramp1) \
      .set('rfsoc/dds0/1/phase', ramp2) \
      .set('rfsoc/dds2/0/amp', 0.1)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(10)],
          'amp': [Pulse(10)],
          'phase': [Pulse(10)]
        },
        1: {
          'freq': [Pulse(10)],
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

@test_utils.with_seq_params
def test_arg_error(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    st = s.add_step(0.01)
    def jasd89jfkalsdfasd():
        st.pulse('rfsoc/dds0/1/amp', 0.2, sth=0.1)
    jasd89jfkalsdfasd()
    with pytest.raises(ValueError,
                       match="Invalid output keyword argument {'sth': 0.1}") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'jasd89jfkalsdfasd')

@test_utils.with_seq_params
def test_ff_ramp(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    st = s.add_step(0.01)
    def ja98fas923ncf():
        st.pulse('rfsoc/dds0/1/ff', test_utils.StaticFunction())
    ja98fas923ncf()
    with pytest.raises(ValueError,
                       match="Feed forward control cannot be ramped") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'ja98fas923ncf')

@test_utils.with_seq_params
def test_rampfunc_error(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    st = s.add_step(0.01)
    def jiasd9f89asd():
        st.pulse('rfsoc/dds0/1/freq', test_utils.ErrorSegment())
    jiasd9f89asd()
    comp.finalize()
    with pytest.raises(ValueError, match="JJJLLL---DFFDAFD") as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'jiasd9f89asd')

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    st = s.add_step(0.01)
    def j89asdf():
        st.pulse('rfsoc/dds0/1/freq', test_utils.CustomSegment((-0.1,)))
    j89asdf()
    comp.finalize()
    with pytest.raises(ValueError, match="Segment time cannot be negative") as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'j89asdf')

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    st = s.add_step(0.01)
    def ajs89dfjasvsrsdsfa():
        st.pulse('rfsoc/dds0/1/freq', test_utils.CustomSegment((0,)))
    ajs89dfjasvsrsdsfa()
    comp.finalize()
    with pytest.raises(ValueError,
                       match="Segment time point must monotonically increase") as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'ajs89dfjasvsrsdsfa')

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    st = s.add_step(0.01)
    def ajs89dfjasvsrsdsf2():
        st.pulse('rfsoc/dds0/1/freq', test_utils.CustomSegment((0.005, 0.002)))
    ajs89dfjasvsrsdsf2()
    comp.finalize()
    with pytest.raises(ValueError,
                       match="Segment time point must monotonically increase") as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'ajs89dfjasvsrsdsf2')

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    st = s.add_step(0.01)
    def jas8faslj34ajsdfa8s9():
        st.pulse('rfsoc/dds0/1/freq', test_utils.CustomSegment((0.005, 0.01)))
    jas8faslj34ajsdfa8s9()
    comp.finalize()
    with pytest.raises(ValueError,
                       match="Segment time point must not exceed action length") as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'jas8faslj34ajsdfa8s9')

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    st = s.add_step(0.01)
    def jaksdjf8a9sdfjas():
        st.pulse('rfsoc/dds0/1/freq', test_utils.CustomSegment(([],)))
    jaksdjf8a9sdfjas()
    comp.finalize()
    with pytest.raises(TypeError) as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'jaksdjf8a9sdfjas')

@test_utils.with_seq_params
def test_val_error(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    # This causes a error to be thrown when converting to boolean
    def j893ajjaks988394():
        s.set('rfsoc/dds0/0/ff', np.array([1, 2]))
    j893ajjaks988394()
    with pytest.raises(ValueError) as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'j893ajjaks988394')

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    # This causes a error to be thrown when converting to float
    def a88f2398fasd():
        s.set('rfsoc/dds3/0/freq', [1, 2])
    a88f2398fasd()
    with pytest.raises(TypeError) as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'a88f2398fasd')

    comp = test_env.new_comp(max_bt)
    s = comp.seq
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
    test_utils.check_bt(exc, max_bt, 'js89j308joro82qwe')

    comp = test_env.new_comp(max_bt)
    s = comp.seq
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
    test_utils.check_bt(exc, max_bt, 'e083jafd')

    def error_callback():
        raise ValueError("AAABBBCCC")

    comp = test_env.new_comp(max_bt)
    s = comp.seq
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
    test_utils.check_bt(exc, max_bt, 'oqo8we9813fasd')

    comp = test_env.new_comp(max_bt)
    s = comp.seq
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
    test_utils.check_bt(exc, max_bt, 'q9e8uasdfasd')

@test_utils.with_seq_params
def test_ff_output(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/ff', False)
    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/ff', True)
    s.add_step(1e-3) \
      .pulse('rfsoc/dds0/0/ff', True)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(409600), Pulse(819208, ff=True)],
          'amp': [Pulse(1228808)],
          'phase': [Pulse(1228808)]
        },
        1: {
          'freq': [Pulse(1228808)],
          'amp': [Pulse(1228808)],
          'phase': [Pulse(1228808)]
        },
    })

    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/ff', False)
    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/ff', True)
    s.add_step(1e-3) \
      .pulse('rfsoc/dds0/0/ff', True, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(409600), Pulse(409600, ff=True),
                   Pulse(409608, sync=True, ff=True)],
          'amp': [Pulse(1228808)],
          'phase': [Pulse(1228808)]
        },
        1: {
          'freq': [Pulse(1228808)],
          'amp': [Pulse(1228808)],
          'phase': [Pulse(1228808)]
        },
    })

@test_utils.with_seq_params
def test_param_output(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/amp', 0)
    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/amp', 0.2)
    s.add_step(1e-3) \
      .pulse('rfsoc/dds0/0/amp', 0.2)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(1228808)],
          'amp': [Pulse(409600), Pulse(819208, Spline(0.2))],
          'phase': [Pulse(1228808)]
        },
        1: {
          'freq': [Pulse(1228808)],
          'amp': [Pulse(1228808)],
          'phase': [Pulse(1228808)]
        },
    })

    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/amp', 0)
    s.add_step(1e-3) \
      .set('rfsoc/dds0/0/amp', 0.2)
    s.add_step(1e-3) \
      .pulse('rfsoc/dds0/0/amp', 0.2, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(819200), Pulse(409608, sync=True)],
          'amp': [Pulse(409600), Pulse(819208, Spline(0.2))],
          'phase': [Pulse(1228808)]
        },
        1: {
          'freq': [Pulse(1228808)],
          'amp': [Pulse(1228808)],
          'phase': [Pulse(1228808)]
        },
    })

@test_utils.with_seq_params
def test_sync_merge(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(0)
    s.set('rfsoc/dds0/0/freq', 120e6, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
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

    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(100e-12)
    s.set('rfsoc/dds0/0/freq', 120e6, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
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

    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(2e-9)
    s.set('rfsoc/dds0/0/freq', 120e6, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(4, Spline(120e6), sync=True), Pulse(5, Spline(120e6))],
          'amp': [Pulse(9)],
          'phase': [Pulse(9)]
        },
        1: {
          'freq': [Pulse(9)],
          'amp': [Pulse(9)],
          'phase': [Pulse(9)]
        },
    })

    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(0)
    s.set('rfsoc/dds0/0/freq', 120e6)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(4, Spline(100e6, 110e6, -180e6, 90e6), sync=True),
                   Pulse(4, Spline(120e6))],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
        1: {
          'freq': [Pulse(8)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
    })

    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(100e-12)
    s.set('rfsoc/dds0/0/freq', 120e6)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(4, Spline(100e6, 110e6, -180e6, 90e6), sync=True),
                   Pulse(4, Spline(120e6))],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
        1: {
          'freq': [Pulse(8)],
          'amp': [Pulse(8)],
          'phase': [Pulse(8)]
        },
    })

    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(2e-9)
    s.set('rfsoc/dds0/0/freq', 120e6)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(4, Spline(100e6, 110e6, -180e6, 90e6), sync=True),
                   Pulse(5, Spline(120e6))],
          'amp': [Pulse(9)],
          'phase': [Pulse(9)]
        },
        1: {
          'freq': [Pulse(9)],
          'amp': [Pulse(9)],
          'phase': [Pulse(9)]
        },
    })

@test_utils.with_seq_params
def test_dyn_seq1(max_bt):
    b1 = True
    v1 = 0.001

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.add_step(0.0005) \
      .pulse('rfsoc/dds0/0/amp', 0.2)
    s.conditional(rtval.new_extern(lambda: b1)) \
      .add_step(rtval.new_extern(lambda: v1)) \
      .set('rfsoc/dds0/0/amp', 0.1)
    s.set('rfsoc/dds0/0/amp', 0.5)
    s.set('rfsoc/dds0/1/amp', 0.1)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(614408)],
          'amp': [Pulse(204800, Spline(0.2)),
                  Pulse(409600, Spline(0.1)),
                  Pulse(8, Spline(0.5))],
          'phase': [Pulse(614408)]
        },
        1: {
          'freq': [Pulse(614408)],
          'amp': [Pulse(614400), Pulse(8, Spline(0.1))],
          'phase': [Pulse(614408)]
        },
    })

    b1 = False
    v1 = 0.001
    comp.runtime_finalize(2)
    test_env.check_output({
        0: {
          'freq': [Pulse(204808)],
          'amp': [Pulse(204800, Spline(0.2)), Pulse(8, Spline(0.5))],
          'phase': [Pulse(204808)]
        },
        1: {
          'freq': [Pulse(204808)],
          'amp': [Pulse(204800), Pulse(8, Spline(0.1))],
          'phase': [Pulse(204808)]
        },
    })

    b1 = True
    v1 = 0
    comp.runtime_finalize(3)
    test_env.check_output({
        0: {
          'freq': [Pulse(204808)],
          'amp': [Pulse(204800, Spline(0.2)), Pulse(8, Spline(0.5))],
          'phase': [Pulse(204808)]
        },
        1: {
          'freq': [Pulse(204808)],
          'amp': [Pulse(204800), Pulse(8, Spline(0.1))],
          'phase': [Pulse(204808)]
        },
    })

    b1 = False
    v1 = 0
    comp.runtime_finalize(4)
    test_env.check_output({
        0: {
          'freq': [Pulse(204808)],
          'amp': [Pulse(204800, Spline(0.2)), Pulse(8, Spline(0.5))],
          'phase': [Pulse(204808)]
        },
        1: {
          'freq': [Pulse(204808)],
          'amp': [Pulse(204800), Pulse(8, Spline(0.1))],
          'phase': [Pulse(204808)]
        },
    })

@test_utils.with_seq_params
def test_tight_output1(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
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
    test_env.check_output({
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
          'phase': [Pulse(12)]
        },
        5: {
          'freq': [Pulse(12)],
          'amp': [Pulse(12)],
          'phase': [Pulse(12)]
        },
    })

@test_utils.with_seq_params
def test_tight_output2(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
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
    test_env.check_output({
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
          'phase': [Pulse(12)]
        },
        5: {
          'freq': [Pulse(12)],
          'amp': [Pulse(12)],
          'phase': [Pulse(12)]
        },
    })

@test_utils.with_seq_params
def test_tight_output3(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
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
    test_env.check_output({
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
          'phase': [Pulse(12)]
        },
        5: {
          'freq': [Pulse(12)],
          'amp': [Pulse(12)],
          'phase': [Pulse(12)]
        },
    })

@test_utils.with_seq_params
def test_tight_output4(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
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
    test_env.check_output({
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
          'phase': [Pulse(12)]
        },
        5: {
          'freq': [Pulse(12)],
          'amp': [Pulse(12)],
          'phase': [Pulse(12)]
        },
    })

@test_utils.with_seq_params
def test_tight_output5(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
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
    test_env.check_output({
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
          'phase': [Pulse(12)]
        },
        5: {
          'freq': [Pulse(12)],
          'amp': [Pulse(12)],
          'phase': [Pulse(12)]
        },
    })

@test_utils.with_seq_params
def test_dds_delay_rt_error(max_bt):
    comp = test_env.new_comp(max_bt)
    comp.rb.set_dds_delay(0, rtval.new_extern(lambda: -0.001))
    comp.finalize()
    with pytest.raises(ValueError, match="DDS time offset -0.001 cannot be negative."):
        comp.runtime_finalize(1)

    comp = test_env.new_comp(max_bt)
    comp.rb.set_dds_delay(1, rtval.new_extern(lambda: 1))
    comp.finalize()
    with pytest.raises(ValueError,
                       match="DDS time offset 1.0 cannot be more than 100ms."):
        comp.runtime_finalize(1)

@test_utils.with_seq_params
@pytest.mark.parametrize('use_rt', [False, True])
def test_dds_delay(max_bt, use_rt):
    def wrap_value(v):
        if use_rt:
            return rtval.new_extern(lambda: v)
        return v
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    with pytest.raises(ValueError, match="DDS time offset -0.001 cannot be negative."):
        comp.rb.set_dds_delay(0, -0.001)
    with pytest.raises(ValueError,
                       match="DDS time offset 1.0 cannot be more than 100ms."):
        comp.rb.set_dds_delay(1, 1)

    comp.rb.set_dds_delay(1, wrap_value(1e-3))
    comp.rb.set_dds_delay(0, wrap_value(1e-6))
    s.add_step(1e-3) \
      .pulse('rfsoc/dds0/0/amp', 0.1) \
      .set('rfsoc/dds0/1/phase', 0.2) \
      .set('rfsoc/dds1/1/freq', 100e6) \
      .pulse('rfsoc/dds2/0/amp', 0.3)
    comp.finalize()

    if not use_rt:
        channels = comp.get_channel_info()
        assert channels.dds_delay == {
            0: 1000_000,
            1: 1000_000_000
        }

    comp.runtime_finalize(1)
    if use_rt:
        channels = comp.get_channel_info()
        assert channels.dds_delay == {
            0: 1000_000,
            1: 1000_000_000
        }
    test_env.check_output({
        0: {
          'freq': [Pulse(819208)],
          'amp': [Pulse(410), Pulse(409600, Spline(0.1)), Pulse(409198)],
          'phase': [Pulse(819208)]
        },
        1: {
          'freq': [Pulse(819208)],
          'amp': [Pulse(819208)],
          'phase': [Pulse(410), Pulse(818798, Spline(0.2))]
        },
        2: {
          'freq': [Pulse(819208)],
          'amp': [Pulse(819208)],
          'phase': [Pulse(819208)]
        },
        3: {
          'freq': [Pulse(409600), Pulse(409608, Spline(100e6))],
          'amp': [Pulse(819208)],
          'phase': [Pulse(819208)]
        },
        4: {
          'freq': [Pulse(819208)],
          'amp': [Pulse(409600, Spline(0.3)), Pulse(409608)],
          'phase': [Pulse(819208)]
        },
        5: {
          'freq': [Pulse(819208)],
          'amp': [Pulse(819208)],
          'phase': [Pulse(819208)]
        },
    })

@test_utils.with_seq_params
def test_cond_ramp_error(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.conditional(rtval.new_extern(lambda: False)) \
      .add_step(rtval.new_extern(lambda: 0)) \
      .set('rfsoc/dds0/0/amp', test_utils.DivLengthFunction()) \
      .pulse('rfsoc/dds0/1/amp', test_utils.DivLengthFunction())
    comp.finalize()
    comp.runtime_finalize(1)

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.add_step(0) \
      .set('rfsoc/dds0/0/amp', Blackman(1)) \
      .pulse('rfsoc/dds0/1/amp', Blackman(1)) \
      .set('rfsoc/dds1/0/amp', BlackmanSquare(1)) \
      .pulse('rfsoc/dds1/1/amp', BlackmanSquare(1))
    comp.finalize()
    comp.runtime_finalize(1)

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.add_step(rtval.new_extern(lambda: 0)) \
      .set('rfsoc/dds0/0/amp', Blackman(1)) \
      .pulse('rfsoc/dds0/1/amp', Blackman(1)) \
      .set('rfsoc/dds1/0/amp', BlackmanSquare(1)) \
      .pulse('rfsoc/dds1/1/amp', BlackmanSquare(1))
    comp.finalize()
    comp.runtime_finalize(1)

@test_utils.with_seq_params
def test_cross_channel_sync1(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
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
    test_env.check_output({
        0: {
          'freq': [Pulse(4096008)],
          'amp': [Pulse(4096008)],
          'phase': [Pulse(4096008)]
        },
        1: {
          'freq': [Pulse(4096000, Spline(100e6, 20e6)), Pulse(8, Spline(120e6))],
          'amp': [Pulse(4096008)],
          'phase': [Pulse(4096008)]
        },
    })

    comp = test_env.new_comp(max_bt)
    s = comp.seq
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
    test_env.check_output({
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
          'amp': [Pulse(4096008)],
          'phase': [Pulse(4096008)]
        },
    })

@test_utils.with_seq_params
def test_cross_channel_sync2(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
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
    test_env.check_output({
        0: {
          'freq': [Pulse(4104)],
          'amp': [Pulse(4104)],
          'phase': [Pulse(4104)]
        },
        1: {
          'freq': [Pulse(2048), Pulse(2056, Spline(110e6), sync=True)],
          'amp': [Pulse(4104)],
          'phase': [Pulse(4104)]
        },
    })

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    @s.add_step
    def freq_step(s):
        s.wait(5e-6)
        s.set('rfsoc/dds0/1/freq', 100e6)
        s.add_step(5e-6) \
          .set('rfsoc/dds0/1/phase', 0.0, sync=True) \
          .set('rfsoc/dds0/1/freq', 110e6)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(4104)],
          'amp': [Pulse(4104)],
          'phase': [Pulse(4104)]
        },
        1: {
          'freq': [Pulse(2048), Pulse(2056, Spline(110e6), sync=True)],
          'amp': [Pulse(4104)],
          'phase': [Pulse(4104)]
        },
    })

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    @s.add_step
    def freq_step(s):
        s.wait(5e-6)
        s.set('rfsoc/dds0/1/freq', 100e6)
        s.set('rfsoc/dds0/1/ff', False, sync=True)
        s.add_step(5e-6) \
          .set('rfsoc/dds0/1/freq', 110e6)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: {
          'freq': [Pulse(4104)],
          'amp': [Pulse(4104)],
          'phase': [Pulse(4104)]
        },
        1: {
          'freq': [Pulse(2048),
                   Pulse(4, Spline(100e6, 55e6, -90e6, 45e6), sync=True),
                   Pulse(2052, Spline(110e6))],
          'amp': [Pulse(4104)],
          'phase': [Pulse(4104)]
        },
    })

@test_utils.with_seq_params
def test_use_all_channels(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    assert not comp.rb.use_all_channels
    comp.rb.use_all_channels = True
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
    test_env.check_output({
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
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    assert not comp.rb.use_all_channels
    comp.rb.use_all_channels = True
    s.wait(5e-6)
    comp.finalize()

    comp.runtime_finalize(1)
    unused = {
        'freq': [Pulse(2056)],
        'amp': [Pulse(2056)],
        'phase': [Pulse(2056)]
    }
    test_env.check_output({})

@test_utils.with_seq_params
def test_long_wait(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.add_step(5000) \
      .set('rfsoc/dds0/1/freq', 100e6) \
      .set('rfsoc/dds0/1/amp', 0.2)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
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

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.add_step(5000) \
      .set('rfsoc/dds0/1/freq', LinearRamp(50e6, 100e6), sync=True) \
      .set('rfsoc/dds0/1/amp', 0.2)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
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
          'amp': [Pulse(1024000000004, Spline(0.2)),
                  Pulse(1024000000004, Spline(0.2))],
          'phase': [Pulse(1024000000004), Pulse(1024000000004)]
        },
    })

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.add_step(10000) \
      .set('rfsoc/dds0/1/freq', 100e6) \
      .set('rfsoc/dds0/1/amp', 0.2)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
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
