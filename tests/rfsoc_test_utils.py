#

from brassboard_seq_rfsoc_backend_utils import *

from brassboard_seq.action import _RampFunctionBase
from brassboard_seq.config import Config
from brassboard_seq import backend, rtval, seq
from brassboard_seq.rfsoc_backend import RFSOCBackend

import py_test_utils as test_utils

import pytest
import copy

class Compiler(backend.SeqCompiler):
    def __init__(self, s, rb):
        super().__init__(s)
        self.rb = rb

    def get_channel_info(self):
        s = self.seq
        channels = get_channel_info(self.rb)
        compiled_info = get_compiled_info(self.rb)
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
        for (chn, actions) in enumerate(test_utils.compiler_get_all_actions(self)):
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

class Env:
    def __init__(self, gen):
        self.gen = gen
        self.conf = Config()
        self.conf.add_supported_prefix('artiq')
        self.conf.add_supported_prefix('rfsoc')

    def new_comp(self, *args):
        s = seq.Seq(self.conf, *args)
        comp = Compiler(s, RFSOCBackend(self.gen))
        comp.add_backend('rfsoc', comp.rb)
        comp.add_backend('artiq', backend.Backend()) # Dummy backend
        return comp

def Spline(order0=0.0, order1=0.0, order2=0.0, order3=0.0):
    return [order0, order1, order2, order3]

def Pulse(cycles, spline=Spline(), sync=False, ff=False):
    return dict(cycles=cycles, spline=spline, sync=sync, ff=ff)

def approx_pulses(param, ps):
    if param == 'freq':
        _abs = 0.0015
    elif param == 'amp':
        _abs = 1.5e-5
    elif param == 'phase' or param == 'frame_rot':
        _abs = 2e-12
    else:
        assert False
    def approx_pulse(p):
        p = copy.copy(p)
        p['spline'] = pytest.approx(p['spline'], abs=_abs, rel=1e-9)
        return p
    return [approx_pulse(p) for p in ps]
