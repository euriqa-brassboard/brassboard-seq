#

import dummy_pulse_compiler
dummy_pulse_compiler.inject()

from brassboard_seq.action import Blackman, BlackmanSquare, LinearRamp, SeqCubicSpline
from brassboard_seq import rfsoc_backend, rtval
import py_test_utils as test_utils
from rfsoc_test_utils import Env as _Env, Spline

import pytest
import typing
import numpy as np

class Tone(typing.NamedTuple):
    cycles: int
    freq: tuple = Spline()
    amp: tuple = Spline()
    phase: tuple = Spline()
    sync: bool = False
    feedback: bool = False

class Env(_Env):
    def __init__(self):
        super().__init__(rfsoc_backend.PulseCompilerGenerator())

    def get_output(self):
        output = self.gen.output
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
                assert not tonedata.frame_rotate_at_end
                assert not tonedata.reset_frame

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

    def check_output(self, expected):
        pulses = self.get_output()
        assert pulses == expected

test_env = Env()

def test_generator():
    rfsoc_backend.PulseCompilerGenerator()
    rfsoc_backend.PulseCompilerGenerator()

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
    test_env.check_output({
        0: [Tone(8)],
        1: [Tone(8)],
        2: [Tone(8)],
        3: [Tone(8)],
        4: [Tone(8)],
        5: [Tone(8)],
    })

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    assert s.get_channel_id('rfsoc/dds0/0') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds0/0'):
        comp.finalize()

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    assert s.get_channel_id('rfsoc/ddsaa/0/freq') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/ddsaa/0/freq'):
        comp.finalize()

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    assert s.get_channel_id('rfsoc/dds0/l/freq') == 0
    with pytest.raises(ValueError, match='Invalid channel name rfsoc/dds0/l/freq'):
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
        0: [Tone(4096008)],
        1: [Tone(4096000, Spline(100e6), Spline(0.2),
                 Spline(0.2 * np.pi), False, True),
            Tone(8, Spline(100e6), Spline(), Spline(0.2 * np.pi), False, True)],
    })

@test_utils.with_seq_params
def test_output2(max_bt):
    b1 = True
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.conditional(test_utils.new_extern(lambda: b1)) \
      .add_step(test_utils.new_extern(lambda: 0.01)) \
      .pulse('rfsoc/dds0/1/amp', test_utils.new_extern(lambda: 0.2)) \
      .set('rfsoc/dds0/1/freq', test_utils.new_extern(lambda: 100e6), sync=1) \
      .set('rfsoc/dds0/1/phase', test_utils.new_extern(lambda: 0.1)) \
      .set('rfsoc/dds0/1/ff', test_utils.new_extern(lambda: True), sync=False)
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
        0: [Tone(4096008)],
        1: [Tone(4096000, Spline(100e6), Spline(0.2),
                 Spline(0.2 * np.pi), True, True),
            Tone(8, Spline(100e6), Spline(), Spline(0.2 * np.pi), False, True)],
    })

    b1 = False
    comp.runtime_finalize(2)
    test_env.check_output({
        0: [Tone(8)],
        1: [Tone(8)],
    })

@test_utils.with_seq_params
def test_output3(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.conditional(False) \
      .add_step(test_utils.new_extern(lambda: 0.01)) \
      .pulse('rfsoc/dds0/1/amp', test_utils.new_extern(lambda: 0.2)) \
      .set('rfsoc/dds0/1/freq', test_utils.new_extern(lambda: 100e6), sync=1) \
      .set('rfsoc/dds0/1/phase', test_utils.new_extern(lambda: 0.1)) \
      .set('rfsoc/dds0/1/ff', test_utils.new_extern(lambda: True), sync=False)
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
        0: [Tone(8)],
        1: [Tone(8)],
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
        0: [Tone(4096008)],
        1: [Tone(2048000, Spline(), pytest.approx(Spline(-0.01, 0.0025)),
                 pytest.approx(Spline(-0.02 * np.pi, 0.005 * np.pi))),
            Tone(2048000, Spline(), pytest.approx(Spline(-0.0075, 0.0025)),
                 pytest.approx(Spline(-0.015 * np.pi, 0.005 * np.pi))),
            Tone(8, Spline(), Spline(), pytest.approx(Spline(-0.01 * np.pi)))],
    })

@test_utils.with_seq_params
def test_ramp_output2(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    ramp1 = test_utils.StaticFunction()
    ramp2 = test_utils.StaticFunction()
    s.add_step(test_utils.new_extern(lambda: 0.01)) \
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
        0: [Tone(4096008)],
        1: [Tone(2048000, Spline(), pytest.approx(Spline(-0.01, 0.0025)),
                 pytest.approx(Spline(-0.02 * np.pi, 0.005 * np.pi))),
            Tone(2048000, Spline(), pytest.approx(Spline(-0.0075, 0.0025)),
                 pytest.approx(Spline(-0.015 * np.pi, 0.005 * np.pi))),
            Tone(8, Spline(), Spline(), pytest.approx(Spline(-0.01 * np.pi)))],
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
        0: [Tone(2048008)],
        1: [Tone(1024000, Spline(), pytest.approx(Spline(0.1, 0.05, 0.0, 0.0))),
            Tone(1024000, Spline(), pytest.approx(Spline(0.15, 0.05, 0.0, 0.0))),
            Tone(8)],
        4: [Tone(2048008, Spline(), Spline(0.1))],
        5: [Tone(2048000, Spline(), Spline(),
                 pytest.approx(Spline(0.2 * np.pi, 0.2 * np.pi))),
            Tone(8, Spline(), Spline(), Spline(0.4 * np.pi))],
        8: [Tone(2048008)],
        9: [Tone(2048000, Spline(), Spline(0.2, 0.1, 0.9, 0.3)),
            Tone(8, Spline(), pytest.approx(Spline(1.5)))],
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
        0: [Tone(2048008)],
        1: [Tone(1024000, Spline(), Spline(),
                 pytest.approx(Spline(0.2 * np.pi, 0.1 * np.pi))),
            Tone(1024000, Spline(), Spline(),
                 pytest.approx(Spline(0.3 * np.pi, 0.1 * np.pi))),
            Tone(8)],
        4: [Tone(2048008, Spline(), Spline(0.1), Spline(-1.0))],
        5: [Tone(2048000, Spline(), Spline(),
                 pytest.approx(Spline(0.4 * np.pi, 0.2 * np.pi, 1.8 * np.pi, 0.6 * np.pi))),
            Tone(8, Spline(), Spline(), pytest.approx(Spline(3 * np.pi)))],
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
        0: [Tone(2048008, Spline(), Spline(0.2), Spline(-0.5))],
        1: [Tone(512000, Spline(), pytest.approx(Spline(0.2, 0.3)),
                 pytest.approx(Spline(1.8 * np.pi, -0.4 * np.pi))),
            Tone(1024000, Spline(), pytest.approx(Spline(0.5, -0.2)),
                 pytest.approx(Spline(1.4 * np.pi, -0.8 * np.pi))),
            Tone(512000, Spline(), pytest.approx(Spline(0.3, -0.1)),
                 pytest.approx(Spline(0.6 * np.pi, 1.2 * np.pi))),
            Tone(8, Spline(), Spline(0.2), Spline(1.8 * np.pi))],
        4: [Tone(1024000, pytest.approx(Spline(80e6, 20e6), abs=1e-3)),
            Tone(1024000, pytest.approx(Spline(100e6, -20e6), abs=1e-3)),
            Tone(8, Spline(80e6))],
        5: [Tone(2048008)],
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
        0: [
            # amp 0.0
            Tone(409600, Spline(),
                 pytest.approx(Spline(0.2, -0.025, 0.05625, -0.0046875))),
            # amp 0.1
            # freq 0.0
            Tone(409600,
                 pytest.approx(Spline(100e6, 2.5e6, -1.25e6, 0.3125e6)),
                 pytest.approx(Spline(0.2265625, 0.0734375, 0.0421875, -0.0046875))),
            # amp 0.2
            # freq 0.1
            # phase 0.0
            Tone(409600,
                 pytest.approx(Spline(101.5625e6, 0.9375e6, -0.3125e6, 0.3125e6)),
                 pytest.approx(Spline(0.3375, 0.14375, 0.028125, -0.0046875)),
                 pytest.approx(Spline(0.8 * np.pi, -0.1 * np.pi,
                                      0.0125 * np.pi, -0.00625 * np.pi))),
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
                                      -0.071875 * np.pi, -0.028125 * np.pi))),
            # freq 1.3
            # phase 1.2
            Tone(409600,
                 pytest.approx(Spline(59.6875e6, 5.3125e6, -4.6875e6, -0.3125e6)),
                 Spline(0.7),
                 pytest.approx(Spline(0.925 * np.pi, 0.0625 * np.pi,
                                      -0.15625 * np.pi, -0.028125 * np.pi))),
            # phase 1.3
            Tone(409600, Spline(60e6), Spline(0.7),
                 pytest.approx(Spline(0.803125 * np.pi, -0.334375 * np.pi,
                                      -0.240625 * np.pi, -0.028125 * np.pi))),
            Tone(409608, Spline(60e6), Spline(0.7),
                 pytest.approx(Spline(0.2 * np.pi)))],
        1: [Tone(4505608)],
        4: [Tone(409600),
            Tone(4096008, Spline(), Spline(0.1))],
        5: [Tone(4505608)],
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
        0: [
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.0, 0.00015745877326822644,
                                      0.01300285490625988, 0.001468462559752276))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.014628776239280439, 0.03074418247908131,
                                      0.01795439872470489, 0.003119251963659586))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.06644660940672625, 0.07585724657247933,
                                      0.028056006241789055, 0.0017298791015365467))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.17208974132253135, 0.13669965899911896,
                                      0.033698736281407626, -0.002488136603058244))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.34, 0.19610073763144964,
                                      0.026062367723541868, -0.007389931667371563))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.5547731736876211, 0.22576257627864393,
                                      0.003121741649712906, -0.010104101022703649))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.7735533905932739, 0.20184724575694735,
                                      -0.028177599439071166, -0.008714728160579277))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.9385083087505672, 0.11992457534234635,
                                      -0.05499955158835057, -0.0034333325045636798))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(1.0, 0.00037452534804338633,
                                      -0.06529954910203895, 0.003433332504564568))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.9385083087505672, -0.11934786239705719,
                                      -0.05432178392081566, 0.008714728160582386))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.7735533905932739, -0.2016937565099568,
                                      -0.027190561418399817, 0.010104101022704093))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.5547731736876214, -0.22605567807641802,
                                      0.0038925727214200734, 0.007389931667376892))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.34, -0.19663272175275923,
                                      0.026234326472228675, 0.002488136603061575))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.17208974132253121, -0.13715889636066686,
                                      0.03324564354639803, -0.0017298791015359916))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.06644660940672625, -0.07601073581946988,
                                      0.02731215461568337, -0.0031192519636593086))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.014628776239280494, -0.03056855626504512,
                                      0.01740824258551721, -0.0014684625597525258))),
            Tone(8, Spline(), pytest.approx(Spline()))
        ],
        1: [Tone(409600008, Spline(), Spline(1.0))]
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
        0: [Tone(2048000, Spline(), pytest.approx(Spline(0.1, 0.4, -0.2, 0.1))),
            Tone(4096000, Spline(), pytest.approx(Spline(0.1, 0.4, -0.2, 0.1))),
            Tone(8, Spline(), Spline(0.4))],
        1: [Tone(6144008)],
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
        0: [Tone(1024000, Spline(), pytest.approx(Spline(0, 0.9))),
            Tone(1024000, Spline(), pytest.approx(Spline(0.9, -0.9))),
            Tone(2048000, Spline(), pytest.approx(Spline(0.3, 0.6))),
            Tone(2048000, Spline(), pytest.approx(Spline(0.9, -0.6))),
            Tone(8, Spline(), Spline(0.3))],
        1: [Tone(6144008)],
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
        0: [Tone(2048000, Spline(), pytest.approx(Spline(0.1, 0.8))),
            Tone(2048000, Spline(), pytest.approx(Spline(0.1, 0.4))),
            Tone(2048000, Spline(), pytest.approx(Spline(0.5, 0.4))),
            Tone(8, Spline(), Spline(0.9))],
        1: [Tone(1024000, Spline(), pytest.approx(Spline(0.1, 0.4))),
            Tone(1024000, Spline(), pytest.approx(Spline(0.5, 0.4))),
            Tone(4096000, Spline(), pytest.approx(Spline(0.1, 0.8))),
            Tone(8, Spline(), Spline(0.9))],
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
        0: [
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.0, 0.00015745877326822644,
                                      0.01300285490625988, 0.001468462559752276))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.014628776239280439, 0.03074418247908131,
                                      0.01795439872470489, 0.003119251963659586))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.06644660940672625, 0.07585724657247933,
                                      0.028056006241789055, 0.0017298791015365467))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.17208974132253135, 0.13669965899911896,
                                      0.033698736281407626, -0.002488136603058244))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.34, 0.19610073763144964,
                                      0.026062367723541868, -0.007389931667371563))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.5547731736876211, 0.22576257627864393,
                                      0.003121741649712906, -0.010104101022703649))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.7735533905932739, 0.20184724575694735,
                                      -0.028177599439071166, -0.008714728160579277))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.9385083087505672, 0.11992457534234635,
                                      -0.05499955158835057, -0.0034333325045636798))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(1.0, 0.00037452534804338633,
                                      -0.06529954910203895, 0.003433332504564568))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.9385083087505672, -0.11934786239705719,
                                      -0.05432178392081566, 0.008714728160582386))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.7735533905932739, -0.2016937565099568,
                                      -0.027190561418399817, 0.010104101022704093))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.5547731736876214, -0.22605567807641802,
                                      0.0038925727214200734, 0.007389931667376892))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.34, -0.19663272175275923,
                                      0.026234326472228675, 0.002488136603061575))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.17208974132253121, -0.13715889636066686,
                                      0.03324564354639803, -0.0017298791015359916))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.06644660940672625, -0.07601073581946988,
                                      0.02731215461568337, -0.0031192519636593086))),
            Tone(25600000, Spline(),
                 pytest.approx(Spline(0.014628776239280494, -0.03056855626504512,
                                      0.01740824258551721, -0.0014684625597525258))),
            Tone(51200000, Spline(),
                 pytest.approx(Spline(0.0, 0.00015745877326822644,
                                      0.01300285490625988, 0.001468462559752276))),
            Tone(51200000, Spline(),
                 pytest.approx(Spline(0.014628776239280439, 0.03074418247908131,
                                      0.01795439872470489, 0.003119251963659586))),
            Tone(51200000, Spline(),
                 pytest.approx(Spline(0.06644660940672625, 0.07585724657247933,
                                      0.028056006241789055, 0.0017298791015365467))),
            Tone(51200000, Spline(),
                 pytest.approx(Spline(0.17208974132253135, 0.13669965899911896,
                                      0.033698736281407626, -0.002488136603058244))),
            Tone(51200000, Spline(),
                 pytest.approx(Spline(0.34, 0.19610073763144964,
                                      0.026062367723541868, -0.007389931667371563))),
            Tone(51200000, Spline(),
                 pytest.approx(Spline(0.5547731736876211, 0.22576257627864393,
                                      0.003121741649712906, -0.010104101022703649))),
            Tone(51200000, Spline(),
                 pytest.approx(Spline(0.7735533905932739, 0.20184724575694735,
                                      -0.028177599439071166, -0.008714728160579277))),
            Tone(51200000, Spline(),
                 pytest.approx(Spline(0.9385083087505672, 0.11992457534234635,
                                      -0.05499955158835057, -0.0034333325045636798))),
            Tone(51200000, Spline(),
                 pytest.approx(Spline(1.0, 0.00037452534804338633,
                                      -0.06529954910203895, 0.003433332504564568))),
            Tone(51200000, Spline(),
                 pytest.approx(Spline(0.9385083087505672, -0.11934786239705719,
                                      -0.05432178392081566, 0.008714728160582386))),
            Tone(51200000, Spline(),
                 pytest.approx(Spline(0.7735533905932739, -0.2016937565099568,
                                      -0.027190561418399817, 0.010104101022704093))),
            Tone(51200000, Spline(),
                 pytest.approx(Spline(0.5547731736876214, -0.22605567807641802,
                                      0.0038925727214200734, 0.007389931667376892))),
            Tone(51200000, Spline(),
                 pytest.approx(Spline(0.34, -0.19663272175275923,
                                      0.026234326472228675, 0.002488136603061575))),
            Tone(51200000, Spline(),
                 pytest.approx(Spline(0.17208974132253121, -0.13715889636066686,
                                      0.03324564354639803, -0.0017298791015359916))),
            Tone(51200000, Spline(),
                 pytest.approx(Spline(0.06644660940672625, -0.07601073581946988,
                                      0.02731215461568337, -0.0031192519636593086))),
            Tone(51200000, Spline(),
                 pytest.approx(Spline(0.014628776239280494, -0.03056855626504512,
                                      0.01740824258551721, -0.0014684625597525258))),
            Tone(8, Spline(), pytest.approx(Spline()))
        ],
        1: [Tone(1228800008, Spline(), Spline(1.0))]
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
        0: [Tone(90)],
        1: [Tone(41, Spline(), pytest.approx(Spline(-2e-7, 5e-8), rel=3e-3),
                 pytest.approx(Spline(-4e-7 * np.pi, 1e-7 * np.pi), rel=3e-3)),
            Tone(41, Spline(), pytest.approx(Spline(-1.5e-7, 5e-8), rel=3e-3),
                 pytest.approx(Spline(-3e-7 * np.pi, 1e-7 * np.pi), rel=3e-3)),
            Tone(8, Spline(), Spline(), pytest.approx(Spline(-2e-7 * np.pi)))],
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
        0: [Tone(10)],
        1: [Tone(4, Spline(),
                 pytest.approx(Spline(0.1, 0.35, -1.35, 0.9), rel=3e-3),
                 pytest.approx(Spline(0.2 * np.pi, 0.5 * np.pi, -0.3 * np.pi, 0),
                               rel=3e-3)),
            Tone(6, Spline(), Spline(), pytest.approx(Spline(0.4 * np.pi)))],
        4: [Tone(10, Spline(), Spline(0.1))],
        5: [Tone(10)],
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
      .pulse('rfsoc/dds1/1/freq', test_utils.new_extern(lambda: 1.23)) \
      .pulse('rfsoc/dds2/0/ff', test_utils.new_extern(lambda: True))
    # This causes a error to be thrown when converting to boolean
    def js89j308joro82qwe():
        s.set('rfsoc/dds0/0/ff', test_utils.new_extern(lambda: np.array([1, 2])))
    js89j308joro82qwe()
    comp.finalize()
    with pytest.raises(TypeError) as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'js89j308joro82qwe')

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.set('rfsoc/dds0/0/ff', True)
    s.add_step(0.01) \
      .pulse('rfsoc/dds1/1/freq', test_utils.new_extern(lambda: 1.23)) \
      .pulse('rfsoc/dds2/0/ff', test_utils.new_extern(lambda: True))
    # This causes a error to be thrown when converting to float
    def e083jafd():
        s.set('rfsoc/dds3/0/amp', test_utils.new_extern(lambda: [1, 2]))
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
      .pulse('rfsoc/dds1/1/freq', test_utils.new_extern(lambda: 1.23)) \
      .pulse('rfsoc/dds2/0/ff', test_utils.new_extern(lambda: True))
    def oqo8we9813fasd():
        s.set('rfsoc/dds0/0/ff', test_utils.new_extern(error_callback))
    oqo8we9813fasd()
    comp.finalize()
    with pytest.raises(ValueError, match="AAABBBCCC") as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'oqo8we9813fasd')

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.set('rfsoc/dds0/0/ff', True)
    s.add_step(0.01) \
      .pulse('rfsoc/dds1/1/freq', test_utils.new_extern(lambda: 1.23)) \
      .pulse('rfsoc/dds2/0/ff', test_utils.new_extern(lambda: True))
    def q9e8uasdfasd():
        s.set('rfsoc/dds3/0/amp', test_utils.new_extern(error_callback))
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
        0: [Tone(409600),
            Tone(819208, feedback=True)],
        1: [Tone(1228808)],
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
        0: [Tone(409600),
            Tone(409600, feedback=True),
            Tone(409608, sync=True, feedback=True)],
        1: [Tone(1228808)],
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
        0: [Tone(409600),
            Tone(819208, Spline(), Spline(0.2))],
        1: [Tone(1228808)],
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
        0: [Tone(409600),
            Tone(409600, Spline(), Spline(0.2)),
            Tone(409608, Spline(), Spline(0.2), sync=True)],
        1: [Tone(1228808)],
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
        0: [Tone(8, Spline(120e6), sync=True)],
        1: [Tone(8)],
    })

    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(100e-12)
    s.set('rfsoc/dds0/0/freq', 120e6, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: [Tone(8, Spline(120e6), sync=True)],
        1: [Tone(8)],
    })

    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(2e-9)
    s.set('rfsoc/dds0/0/freq', 120e6, sync=True)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: [Tone(4, Spline(120e6), sync=True),
            Tone(5, Spline(120e6))],
        1: [Tone(9)],
    })

    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(0)
    s.set('rfsoc/dds0/0/freq', 120e6)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: [Tone(4, Spline(100e6, 110e6, -180e6, 90e6), sync=True),
            Tone(4, Spline(120e6))],
        1: [Tone(8)],
    })

    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(100e-12)
    s.set('rfsoc/dds0/0/freq', 120e6)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: [Tone(4, Spline(100e6, 110e6, -180e6, 90e6), sync=True),
            Tone(4, Spline(120e6))],
        1: [Tone(8)],
    })

    comp = test_env.new_comp(max_bt)
    s = comp.seq

    s.set('rfsoc/dds0/0/freq', 100e6, sync=True)
    s.wait(2e-9)
    s.set('rfsoc/dds0/0/freq', 120e6)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: [Tone(4, Spline(100e6, 110e6, -180e6, 90e6), sync=True),
            Tone(5, Spline(120e6))],
        1: [Tone(9)],
    })

@test_utils.with_seq_params
def test_dyn_seq1(max_bt):
    b1 = True
    v1 = 0.001

    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.add_step(0.0005) \
      .pulse('rfsoc/dds0/0/amp', 0.2)
    s.conditional(test_utils.new_extern(lambda: b1)) \
      .add_step(test_utils.new_extern(lambda: v1)) \
      .set('rfsoc/dds0/0/amp', 0.1)
    s.set('rfsoc/dds0/0/amp', 0.5)
    s.set('rfsoc/dds0/1/amp', 0.1)
    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_output({
        0: [Tone(204800, Spline(), Spline(0.2)),
            Tone(409600, Spline(), Spline(0.1)),
            Tone(8, Spline(), Spline(0.5))],
        1: [Tone(614400),
            Tone(8, Spline(), Spline(0.1))],
    })

    b1 = False
    v1 = 0.001
    comp.runtime_finalize(2)
    test_env.check_output({
        0: [Tone(204800, Spline(), Spline(0.2)),
            Tone(8, Spline(), Spline(0.5))],
        1: [Tone(204800),
            Tone(8, Spline(), Spline(0.1))],
    })

    b1 = True
    v1 = 0
    comp.runtime_finalize(3)
    test_env.check_output({
        0: [Tone(204800, Spline(), Spline(0.2)),
            Tone(8, Spline(), Spline(0.5))],
        1: [Tone(204800),
            Tone(8, Spline(), Spline(0.1))],
    })

    b1 = False
    v1 = 0
    comp.runtime_finalize(4)
    test_env.check_output({
        0: [Tone(204800, Spline(), Spline(0.2)),
            Tone(8, Spline(), Spline(0.5))],
        1: [Tone(204800),
            Tone(8, Spline(), Spline(0.1))],
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
        0: [Tone(12)],
        1: [Tone(4, pytest.approx(Spline(100e6, 110e6, -180e6, 90e6),
                                  rel=1e-5, abs=1e-3),
                 pytest.approx(Spline(0.1, 0.55, -0.9, 0.45)),
                 pytest.approx(Spline(0.4 * np.pi))),
            Tone(8, Spline(200e6), Spline(0.9), pytest.approx(Spline(0.2 * np.pi)))],
        4: [Tone(4, Spline(),
                 pytest.approx(Spline(0.3, -0.8, 3, -1.8)), feedback=True),
            Tone(8, Spline(), Spline(0.3), feedback=True)],
        5: [Tone(12)],
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
        0: [Tone(12)],
        1: [Tone(4, pytest.approx(Spline(200e6, -320e6, 600e6, -360e6),
                                  rel=1e-5, abs=1e-3),
                 pytest.approx(Spline(0.1, 0.55, -0.9, 0.45)),
                 pytest.approx(Spline(0.4 * np.pi)),
                 True),
            Tone(8, Spline(200e6), Spline(0.9), pytest.approx(Spline(0.2 * np.pi)))],
        4: [Tone(4, Spline(),
                 pytest.approx(Spline(0.3, -0.8, 3, -1.8)), feedback=True),
            Tone(8, Spline(), Spline(0.3), feedback=True)],
        5: [Tone(12)],
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
        0: [Tone(12)],
        1: [Tone(4, pytest.approx(Spline(120e6), rel=1e-5, abs=1e-3),
                 pytest.approx(Spline(0.1, 0.55, -0.9, 0.45)),
                 pytest.approx(Spline(0.4 * np.pi)),
                 True),
            Tone(8, Spline(200e6), Spline(0.9), pytest.approx(Spline(0.2 * np.pi)))],
        4: [Tone(4, Spline(),
                 pytest.approx(Spline(0.3, -0.8, 3, -1.8)), feedback=True),
            Tone(8, Spline(), Spline(0.3), feedback=True)],
        5: [Tone(12)],
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
        0: [Tone(12)],
        1: [Tone(4, pytest.approx(Spline(200e6, -320e6, 600e6, -360e6),
                                  rel=1e-5, abs=1e-3),
                 pytest.approx(Spline(0.1, 0.55, -0.9, 0.45)),
                 pytest.approx(Spline(0.4 * np.pi)),
                 True),
            Tone(8, Spline(200e6), Spline(0.9), pytest.approx(Spline(0.2 * np.pi)))],
        4: [Tone(4, Spline(),
                 pytest.approx(Spline(0.3, -0.8, 3, -1.8)), feedback=True),
            Tone(8, Spline(), Spline(0.3), feedback=True)],
        5: [Tone(12)],
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
        0: [Tone(12)],
        1: [Tone(4, pytest.approx(Spline(200e6, -320e6, 600e6, -360e6),
                                  rel=1e-5, abs=1e-3),
                 pytest.approx(Spline(0.1, 0.55, -0.9, 0.45)),
                 pytest.approx(Spline(0.4 * np.pi)),
                 True),
            Tone(8, Spline(200e6), Spline(0.9), pytest.approx(Spline(0.2 * np.pi)))],
        4: [Tone(4, Spline(),
                 pytest.approx(Spline(0.3, -0.8, 3, -1.8)), feedback=True),
            Tone(8, Spline(), Spline(0.3), feedback=True)],
        5: [Tone(12)],
    })

@test_utils.with_seq_params
def test_dds_delay_rt_error(max_bt):
    comp = test_env.new_comp(max_bt)
    comp.rb.set_dds_delay(0, test_utils.new_extern(lambda: -0.001))
    comp.finalize()
    with pytest.raises(ValueError, match="DDS time offset -0.001 cannot be negative."):
        comp.runtime_finalize(1)

    comp = test_env.new_comp(max_bt)
    comp.rb.set_dds_delay(1, test_utils.new_extern(lambda: 1))
    comp.finalize()
    with pytest.raises(ValueError,
                       match="DDS time offset 1.0 cannot be more than 100ms."):
        comp.runtime_finalize(1)

@test_utils.with_seq_params
@pytest.mark.parametrize('use_rt', [False, True])
def test_dds_delay(max_bt, use_rt):
    def wrap_value(v):
        if use_rt:
            return test_utils.new_extern(lambda: v)
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
        0: [Tone(410),
            Tone(409600, Spline(), Spline(0.1)),
            Tone(409198)],
        1: [Tone(410),
            Tone(818798, Spline(), Spline(), Spline(0.4 * np.pi))],
        2: [Tone(819208)],
        3: [Tone(409600),
            Tone(409608, Spline(100e6))],
        4: [Tone(409600, Spline(), Spline(0.3)),
            Tone(409608)],
        5: [Tone(819208)],
    })

@test_utils.with_seq_params
def test_cond_ramp_error(max_bt):
    comp = test_env.new_comp(max_bt)
    s = comp.seq
    s.conditional(test_utils.new_extern(lambda: False)) \
      .add_step(test_utils.new_extern(lambda: 0)) \
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
    s.add_step(test_utils.new_extern(lambda: 0)) \
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
        0: [Tone(4096008)],
        1: [Tone(4096000, pytest.approx(Spline(100e6, 20e6), abs=1e-6)),
            Tone(8, Spline(120e6))],
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
        0: [Tone(4096008)],
        1: [Tone(1024000, pytest.approx(Spline(100e6, 5e6), abs=1e-6)),
            Tone(1024000, pytest.approx(Spline(105e6, 5e6), abs=1e-6), sync=True),
            Tone(1024000, pytest.approx(Spline(110e6, 5e6), abs=1e-6), sync=True),
            Tone(1024000, pytest.approx(Spline(115e6, 5e6), abs=1e-6), sync=True),
            Tone(8, Spline(120e6))],
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
        0: [Tone(4104)],
        1: [Tone(2048),
            Tone(2056, Spline(110e6), sync=True)],
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
        0: [Tone(4104)],
        1: [Tone(2048),
            Tone(2056, Spline(110e6), sync=True)],
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
        0: [Tone(4104)],
        1: [Tone(2048),
            Tone(4, pytest.approx(Spline(100e6, 55e6, -90e6, 45e6)), sync=True),
            Tone(2052, Spline(110e6))],
    })
