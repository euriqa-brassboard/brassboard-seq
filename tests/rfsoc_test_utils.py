#

from brassboard_seq_rfsoc_backend_utils import *

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
        assert test_utils.compiler_num_basic_seq(self) == 1
        for (chn, actions) in enumerate(test_utils.compiler_get_all_actions(self, 0)):
            for action in actions:
                if action.get_cond() is False:
                    continue
                all_actions[action.get_aid()] = (chn, action, [False, False])
        bool_values_used = [False for _ in range(len(compiled_info.bool_values))]
        float_values_used = [False for _ in range(len(compiled_info.float_values))]
        relocations_used = [False for _ in range(len(compiled_info.relocations))]

        for chn in channels.channels:
            for (param, param_actions) in enumerate(chn.actions):
                param = ['freq', 'amp', 'phase', 'ff'][param]
                for rfsoc_action in param_actions:
                    chn, action, seen = all_actions[rfsoc_action.aid]
                    action_info = action.get_compile_info()
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

                    cond = action.get_cond()
                    if cond_idx >= 0:
                        bool_values_used[cond_idx] = True
                        assert isinstance(cond, rtval.RuntimeValue)
                        assert compiled_info.bool_values[cond_idx] is cond
                    else:
                        assert cond

                    event_time = test_utils.seq_get_event_time(s, rfsoc_action.tid, 0)
                    static_time = test_utils.event_time_get_static(event_time)
                    if time_idx >= 0:
                        assert static_time == -1
                    else:
                        assert static_time == rfsoc_action.seq_time

                    action_value = action.get_value()
                    isramp = test_utils.isramp(action_value)
                    if rfsoc_action.tid == action_info['tid']:
                        assert not seen[0]
                        seen[0] = True
                        value = action_value
                        assert isramp == rfsoc_action.isramp
                    else:
                        assert rfsoc_action.tid == action_info['end_tid']
                        isramp = test_utils.isramp(action.get_value())
                        assert action.get_is_pulse() or isramp
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
            action_value = action.get_value()
            isramp = test_utils.isramp(action_value)
            if action.get_is_pulse() or isramp:
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

def _get_expected(v):
    try:
        return v.expected
    except AttributeError:
        return v

def combine_err(e1, e2):
    if e1 is None:
        return e2
    elif e2 is None:
        return e1
    return e1 + e2

def add_approx(v, abs=None, rel=None):
    try:
        v0 = v.expected
        abs = combine_err(abs, v.abs)
        rel = combine_err(rel, v.rel)
        return pytest.approx(v0, abs=abs, rel=rel)
    except AttributeError:
        return pytest.approx(v, abs=abs, rel=rel)

def _guess_shift(spl, scale):
    spl1 = abs(_get_expected(spl[1]) / scale)
    spl2 = abs(_get_expected(spl[2]) / scale)
    spl3 = abs(_get_expected(spl[3]) / scale)
    for shift in range(11, 0, -1):
        if spl3 * 2**(3 * shift) < 0.5 and spl2 * 2**(2 * shift) < 0.5 and spl1 * 2**shift < 0.5:
            return shift
    return 0

def approx_pulses(param, ps):
    if param == 'freq':
        _abs = 0.0015
    elif param == 'amp':
        _abs = 2e-5
    elif param == 'phase' or param == 'frame_rot':
        _abs = 2e-12
    else:
        assert False

    scale = param_scale(param)
    def approx_pulse(p):
        p = copy.copy(p)
        spl = p['spline']
        __abs = 1e-9
        __rel = 1e-9
        try:
            _spl = spl
            spl = _spl.expected
            if _spl.abs is not None:
                __abs += _spl.abs
            if _spl.rel is not None:
                __rel += _spl.rel
        except AttributeError:
            pass
        shift = _guess_shift(spl, scale)
        spl = copy.copy(spl)
        spl[0] = add_approx(spl[0], abs=_abs)
        p['spline'] = approx_spline(param, shift, spl, p['cycles'], True, __abs, __rel)
        return p
    return [approx_pulse(p) for p in ps]

def pad_list(lst, n, v):
    return lst + [v] * (n - len(lst))

def check_spline_shift(shift, isp1, isp2, isp3, may_overflow=False):
    if may_overflow and shift == 0 and (isp1 == 0 or isp2 == 0 or isp3 == 0):
        return
    if shift < 11:
        assert (abs(isp1) >> 38) or (abs(isp2) >> 37) or (abs(isp3) >> 36)

def order_precision(order, shift):
    if order < 2:
        return 2**-40
    elif order == 2:
        return 2**-(40 + min(shift * 2, 16))
    elif order == 3:
        return 2**-(40 + min(shift * 3, 32))

def param_scale(param):
    if param == 'freq':
        return 819.2e6
    elif param == 'amp':
        return 1 - 2**-16
    else:
        return 1.0

def flatten_tol(tol):
    if tol is None:
        return [0, 0, 0, 0]
    if hasattr(tol, '__len__'):
        return pad_list(list(tol), 4, 0)
    return [tol, tol, tol, tol]

def approx_spline(param, shift, target, cycles, approx, _abs, rel):
    target = pad_list(list(target), 4, 0)
    if not approx:
        return target
    _abs = flatten_tol(_abs)
    rel = [r + 2**-40 for r in flatten_tol(rel)]
    scale = param_scale(param)

    abs_prec = [order_precision(order, shift) * scale for order in range(4)]
    abs_prec[1] = (abs_prec[1] + abs_prec[2] / 2 + abs_prec[3] / 3) * cycles
    abs_prec[2] = (abs_prec[2] + abs_prec[3]) / 2 * cycles**2
    abs_prec[3] = abs_prec[3] / 6 * cycles**3

    return [add_approx(target[order], abs=_abs[order] + abs_prec[order], rel=rel[order])
            for order in range(4)]
