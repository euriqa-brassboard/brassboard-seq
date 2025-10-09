#

import dummy_artiq

from brassboard_seq.config import Config
from brassboard_seq import action, artiq_backend, backend, seq, rtval
import py_test_utils as test_utils
import brassboard_seq_artiq_backend_utils as artiq_utils
import pytest
import numpy as np

def bytes_to_int32(i0, i8, i16, i24):
    return i0 | (i8 << 8) | (i16 << 16) | (i24 << 24)

def bytes_to_int64(i0, i8, i16, i24, i32, i40, i48, i56):
    return bytes_to_int32(i0, i8, i16, i24) | bytes_to_int32(i32, i40, i48, i56)

class CompiledInfoChecker:
    def __init__(self, comp):
        self.comp = comp
        self.compiled_info = artiq_utils.get_compiled_info(comp.ab)
        self.channels = artiq_utils.get_channel_info(comp.ab)

    def check(self):
        self.bool_values_used = [False] * len(self.compiled_info.bool_values)
        self.float_values_used = [False] * len(self.compiled_info.float_values)
        self.relocations_used = [False] * len(self.compiled_info.relocations)
        ncbseq = test_utils.compiler_num_basic_seq(self.comp)
        if not self.comp.allow_branch:
            assert ncbseq == 1
        assert len(self.compiled_info.all_actions) == ncbseq
        assert len(self.compiled_info.start_values) == ncbseq
        for cbseq_id in range(ncbseq):
            self.check_cbseq(cbseq_id)
        assert all(self.bool_values_used)
        assert all(self.float_values_used)
        assert all(self.relocations_used)

    def check_cbseq(self, cbseq_id):
        bseq_id = test_utils.compiler_get_bseq_id(self.comp, cbseq_id)
        if cbseq_id == 0:
            assert self.compiled_info.start_values[cbseq_id] == []
        else:
            all_starts = {}
            for (chn, start_value) in enumerate(test_utils.compiler_get_all_start_values(self.comp, cbseq_id)):
                if chn in self.channels.ttl_chn_map:
                    ttlid = self.channels.ttl_chn_map[chn]
                    ttlchn = self.channels.ttlchns[ttlid]
                    key = (ttlid, 'counter' if ttlchn.iscounter else 'ttl')
                else:
                    assert chn in self.channels.dds_param_chn_map
                    key = self.channels.dds_param_chn_map[chn]
                all_starts[key] = (start_value, [False])
            for sv in self.compiled_info.start_values[cbseq_id]:
                value, used = all_starts[(sv.chn_idx, sv.type)]
                used[0] = True
                val_idx = sv.val_id
                if val_idx >= 0:
                    assert isinstance(value, rtval.RuntimeValue)
                    if sv.type == 'ttl' or sv.type == 'counter':
                        assert self.compiled_info.bool_values[val_idx] is value
                        self.bool_values_used[val_idx] = True
                    else:
                        assert self.compiled_info.float_values[val_idx] is value
                        self.float_values_used[val_idx] = True
                else:
                    assert not isinstance(value, rtval.RuntimeValue)
                    if sv.type == 'ttl':
                        assert value == bool(sv.value)
                    elif sv.type == 'counter':
                        assert value == bool(sv.value)
                    elif sv.type == 'ddsfreq':
                        assert dds_freq_to_mu(value) == sv.value
                    elif sv.type == 'ddsamp':
                        assert dds_amp_to_mu(value) == sv.value
                    elif sv.type == 'ddsphase':
                        assert dds_phase_to_mu(value) == sv.value
                    else:
                        assert False, "Unknown channel type"
            assert all(used[0] for (value, used) in all_starts.values())
        all_actions = {}
        for (chn, actions) in enumerate(test_utils.compiler_get_all_actions(self.comp, cbseq_id)):
            for action in actions:
                if action.get_cond() is False:
                    continue
                all_actions[action.get_aid()] = (chn, action, [False, False])
        for artiq_action in self.compiled_info.all_actions[cbseq_id]:
            chn, action, seen = all_actions[artiq_action.aid]
            action_info = action.get_compile_info()
            assert action.get_exact_time() == artiq_action.exact_time
            if artiq_action.reloc_id >= 0:
                reloc = self.compiled_info.relocations[artiq_action.reloc_id]
                self.relocations_used[artiq_action.reloc_id] = True
                assert reloc.cond_idx >= 0 or reloc.val_idx >= 0 or reloc.time_idx >= 0
                cond_idx = reloc.cond_idx
                val_idx = reloc.val_idx
                time_idx = reloc.time_idx
            else:
                cond_idx = -1
                val_idx = -1
                time_idx = -1

            cond = action.get_cond()
            if cond_idx >= 0:
                self.bool_values_used[cond_idx] = True
                assert isinstance(cond, rtval.RuntimeValue)
                assert self.compiled_info.bool_values[cond_idx] is cond
            else:
                assert cond

            event_time = test_utils.seq_get_event_time(self.comp.seq,
                                                       artiq_action.tid, bseq_id)
            static_time = test_utils.event_time_get_static(event_time)
            if time_idx >= 0:
                assert static_time == -1
            else:
                assert seq_time_to_mu(static_time) == artiq_action.time_mu

            if artiq_action.tid == action_info['tid']:
                assert not seen[0]
                seen[0] = True
                value = action.get_value()
            else:
                assert artiq_action.tid == action_info['end_tid']
                assert action.get_is_pulse()
                assert not seen[1]
                seen[1] = True
                value = action_info['end_val']
            # Check channel
            if chn in self.channels.ttl_chn_map:
                is_ttl = True
                ttlid = self.channels.ttl_chn_map[chn]
                assert ttlid == artiq_action.chn_idx
                ttlchn = self.channels.ttlchns[ttlid]
                if ttlchn.iscounter:
                    assert artiq_action.type == 'counter'
                else:
                    assert artiq_action.type == 'ttl'
            else:
                is_ttl = False
                assert chn in self.channels.dds_param_chn_map
                ddsid, chntype = self.channels.dds_param_chn_map[chn]
                assert artiq_action.type == chntype
                assert artiq_action.chn_idx == ddsid
            # Check value
            if val_idx >= 0:
                assert isinstance(value, rtval.RuntimeValue)
                if is_ttl:
                    self.bool_values_used[val_idx] = True
                    assert self.compiled_info.bool_values[val_idx] is value
                else:
                    self.float_values_used[val_idx] = True
                    assert self.compiled_info.float_values[val_idx] is value
            else:
                assert not isinstance(value, rtval.RuntimeValue)
                if artiq_action.type == 'ttl':
                    assert value == bool(artiq_action.value)
                elif artiq_action.type == 'counter':
                    assert value == bool(artiq_action.value)
                elif artiq_action.type == 'ddsfreq':
                    assert dds_freq_to_mu(value) == artiq_action.value
                elif artiq_action.type == 'ddsamp':
                    assert dds_amp_to_mu(value) == artiq_action.value
                elif artiq_action.type == 'ddsphase':
                    assert dds_phase_to_mu(value) == artiq_action.value
                else:
                    assert False, "Unknown channel type"

        for chn, action, seen in all_actions.values():
            assert seen[0]
            if action.get_is_pulse():
                assert seen[1]
            else:
                assert not seen[1]

class Compiler(backend.SeqCompiler):
    def __init__(self, s, ab, allow_branch):
        super().__init__(s)
        self.ab = ab
        self.allow_branch = allow_branch

    def get_channel_info(self):
        channels = artiq_utils.get_channel_info(self.ab)
        ttl_tgts = set(ttl.target for ttl in channels.ttlchns)
        assert len(ttl_tgts) == len(channels.ttlchns)
        assert len(ttl_tgts) == len(channels.ttl_chn_map)
        bus_chns = set(bus.channel for bus in channels.urukul_busses)
        assert len(bus_chns) == len(channels.urukul_busses)
        bus_ids = set(dds.bus_id for dds in channels.ddschns)
        assert bus_ids == set(range(len(bus_chns)))
        all_chip_selects = {i: set() for i in range(len(bus_chns))}
        for dds in channels.ddschns:
            chip_selects = all_chip_selects[dds.bus_id]
            assert dds.chip_select not in chip_selects
            chip_selects.add(dds.chip_select)
        assert sum(len(cs) for cs in all_chip_selects.values()) == len(channels.ddschns)
        return channels

    def get_compiled_info(self):
        checker = CompiledInfoChecker(self)
        checker.check()
        return checker.compiled_info

class Env:
    def __init__(self):
        self.use_dma = False
        self.allow_branch = False
        self.np_rtio = np.ndarray((0,), np.int32)
        self.dma_buf = bytearray()
        self.conf = Config()
        self.conf.add_supported_prefix('artiq')
        self.conf.add_supported_prefix('rfsoc')

    def get_list_bytecode(self, output):
        res = [int(r) for r in output.rtios]
        n = len(res)
        i = 0
        total_mu = 0
        while i < n:
            r = int(res[i])
            if r < 0:
                total_mu -= r
                i += 1
            else:
                assert r != 0x80000000
                i += 2
        assert total_mu == output.total_time_mu
        return res

    def get_list_dma(self, output):
        dma_buf = output.rtios
        assert len(dma_buf) % 64 == 0
        cur_t = 0
        res = []
        def set_cur_time(new_t):
            nonlocal cur_t
            wait_t = new_t - cur_t
            assert wait_t >= 0
            cur_t = new_t
            while wait_t > 0x80000000:
                res.append(-0x80000000)
                wait_t -= 0x80000000
            if wait_t == 0:
                return
            res.append(-wait_t)
        def add_action(t, target, value):
            set_cur_time(t)
            res.append(target)
            res.append(value)
        for i in range(0, len(dma_buf), 17):
            if dma_buf[i] == 0:
                assert len(dma_buf) - i <= 64
                for j in range(i + 1, len(dma_buf)):
                    assert dma_buf[j] == 0
                set_cur_time(output.total_time_mu)
                return res
            assert dma_buf[i] == 17
            target = bytes_to_int32(dma_buf[i + 12], dma_buf[i + 1],
                                    dma_buf[i + 2], dma_buf[i + 3])
            value = bytes_to_int32(dma_buf[i + 13], dma_buf[i + 14],
                                   dma_buf[i + 15], dma_buf[i + 16])
            t = bytes_to_int64(dma_buf[i + 4], dma_buf[i + 5],
                               dma_buf[i + 6], dma_buf[i + 7],
                               dma_buf[i + 8], dma_buf[i + 9],
                               dma_buf[i + 10], dma_buf[i + 11])
            assert t >= 0
            add_action(t, target, value)
        assert False

    def get_lists(self, comp):
        getter = (self.get_list_dma if self.use_dma else self.get_list_bytecode)
        assert comp.ab.output[0].total_time_mu == comp.ab.total_time_mu
        outputs = comp.ab.output
        comp.last_list = [getter(output) for output in outputs]
        assert len(outputs) >= 1
        if not self.allow_branch:
            assert len(outputs) == 1
        if len(outputs) == 1:
            output0 = outputs[0]
            assert output0.bseq_id == 0
            assert output0.may_term
            assert output0.next == []
        return comp.last_list

    def get_list(self, comp, cbseq_id=0):
        return self.get_lists(comp)[cbseq_id]

    def check_lists(self, comp, expected):
        rtio = self.get_lists(comp)
        assert rtio == expected

    def check_list(self, comp, expected, cbseq_id=0):
        rtio = self.get_list(comp, cbseq_id)
        assert rtio == expected

    def check_unchanged(self, comp):
        last_list = comp.last_list
        assert self.get_lists(comp) == last_list

    def new_comp(self, max_bt, sys):
        s = seq.Seq(self.conf, max_bt)
        if self.allow_branch:
            ab = artiq_backend.ArtiqBackend(sys, output_format=('dma' if self.use_dma
                                                                else 'bytecode'))
        elif self.use_dma:
            ab = artiq_backend.ArtiqBackend(sys, self.dma_buf, output_format='dma')
            assert ab.output[0].rtios is self.dma_buf
        else:
            ab = artiq_backend.ArtiqBackend(sys, self.np_rtio)
            assert ab.output[0].rtios is self.np_rtio
        comp = Compiler(s, ab, self.allow_branch)
        comp.add_backend('artiq', ab)
        comp.add_backend('rfsoc', backend.Backend()) # Dummy backend
        return comp

test_env = Env()

def seq_time_to_mu(time):
    return (time + 500) // 1000

def dds_amp_to_mu(amp):
    v = int(amp * 0x3fff + 0.5)
    if v < 0:
        return 0
    if v > 0x3fff:
        return 0x3fff
    return v

def dds_phase_to_mu(phase):
    return round(phase * 0x10000) & 0xffff

def dds_freq_to_mu(freq):
    return int(freq * 4.294967296 + 0.5)

dds_data_addr = 0x0e << 24
def dds_data1(amp, phase):
    return (dds_amp_to_mu(amp) << 16) | dds_phase_to_mu(phase)
def dds_data2(freq):
    return dds_freq_to_mu(freq)

def _dds_config(flags, length, cs):
    SPIT_DDS_WR = 2
    return flags | ((length - 1) << 8) | ((SPIT_DDS_WR - 2) << 16) | (cs << 24)
def _dds_data_len(length):
    SPIT_DDS_WR = 2
    return ((length + 1) * SPIT_DDS_WR + 1) * 8

dds_config_len = 8
def dds_config_addr(cs):
    return _dds_config(0x8, 8, cs)
dds_addr_len = _dds_data_len(8)
def dds_config_data1(cs):
    return _dds_config(0x8, 32, cs)
dds_data_len = _dds_data_len(32)
def dds_config_data2(cs):
    return _dds_config(0xa, 32, cs)

dds_conf_addr_len = dds_config_len + dds_addr_len
dds_conf_data_len = dds_config_len + dds_data_len
dds_total_len = dds_conf_addr_len + dds_conf_data_len * 2
dds_headless_len = dds_conf_addr_len + dds_conf_data_len * 2 - dds_config_len

def with_dma_param(f, require_branch):
    import inspect
    old_sig = inspect.signature(f)
    params = [inspect.Parameter('use_dma', inspect.Parameter.POSITIONAL_OR_KEYWORD),
              inspect.Parameter('allow_branch', inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    params.extend(old_sig.parameters.values())
    new_sig = inspect.Signature(params)
    def cb(use_dma, allow_branch, *args, **kws):
        test_env.use_dma = use_dma
        test_env.allow_branch = allow_branch
        return f(*args, **kws)
    cb.__name__ = f.__name__
    if hasattr(f, 'pytestmark'):
        cb.pytestmark = f.pytestmark
    cb.__signature__ = new_sig
    cb = pytest.mark.parametrize("use_dma", [False, True])(cb)
    if require_branch:
        return pytest.mark.parametrize("allow_branch", [True])(cb)
    return pytest.mark.parametrize("allow_branch", [False, True])(cb)

def with_artiq_params(f):
    f = test_utils.with_seq_params(f)
    return with_dma_param(f, False)

def with_artiq_br_params(f):
    f = test_utils.with_seq_params(f)
    return with_dma_param(f, True)

def test_constructors():
    with pytest.raises(TypeError):
        artiq_backend.ArtiqBackend()
    artiq_backend.ArtiqBackend(dummy_artiq.DummyDaxSystem(),
                               np.ndarray((0,), np.int32))
    artiq_backend.ArtiqBackend(dummy_artiq.DummyDaxSystem(),
                               np.ndarray((0,), np.int32), output_format="bytecode")
    with pytest.raises(TypeError):
        artiq_backend.ArtiqBackend(dummy_artiq.DummyDaxSystem(), [])
    with pytest.raises(TypeError, match="RTIO output must be a int32 array"):
        artiq_backend.ArtiqBackend(dummy_artiq.DummyDaxSystem(),
                                   np.ndarray((0,), np.int64))
    with pytest.raises(ValueError, match="RTIO output must be a 1D array"):
        artiq_backend.ArtiqBackend(dummy_artiq.DummyDaxSystem(),
                                   np.ndarray((2, 2), np.int32))
    artiq_backend.ArtiqBackend(dummy_artiq.DummyDaxSystem(),
                               bytearray(), output_format="dma")
    with pytest.raises(TypeError):
        artiq_backend.ArtiqBackend(dummy_artiq.DummyDaxSystem(),
                                   [], output_format="dma")
    with pytest.raises(TypeError):
        artiq_backend.ArtiqBackend(dummy_artiq.DummyDaxSystem(),
                                   np.ndarray((0,), np.int32), output_format="dma")
    with pytest.raises(ValueError, match="Unknown output type: 'invalid'"):
        artiq_backend.ArtiqBackend(dummy_artiq.DummyDaxSystem(),
                                   [], output_format="invalid")

@with_artiq_params
def test_val_error(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    # This causes a error to be thrown when converting to boolean
    def ajquoeiauhfasd():
        s.set('artiq/ttl0', np.array([1, 2]))
    ajquoeiauhfasd()
    with pytest.raises(ValueError) as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'ajquoeiauhfasd')

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    # This causes a error to be thrown when converting to float
    def jaus7hfas9dafs():
        s.set('artiq/urukul3_ch0/freq', [1, 2])
    jaus7hfas9dafs()
    with pytest.raises(TypeError) as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'jaus7hfas9dafs')

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.set('artiq/ttl0', True)
    s.add_step(0.01) \
      .pulse('artiq/urukul0_ch2/freq', test_utils.new_extern(lambda: 1.23)) \
      .pulse('artiq/ttl2', test_utils.new_extern(lambda: True))
    # This causes a error to be thrown when converting to boolean
    def ajquo1827uhfasd():
        s.set('artiq/ttl0', test_utils.new_extern(lambda: np.array([1, 2])))
    ajquo1827uhfasd()
    comp.finalize()
    with pytest.raises(TypeError) as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'ajquo1827uhfasd')

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.set('artiq/ttl0', True)
    s.add_step(0.01) \
      .pulse('artiq/urukul0_ch2/freq', test_utils.new_extern(lambda: 1.23)) \
      .pulse('artiq/ttl2', test_utils.new_extern(lambda: True))
    # This causes a error to be thrown when converting to float
    def jausasdjklfa834fs():
        s.set('artiq/urukul3_ch0/amp', test_utils.new_extern(lambda: [1, 2]))
    jausasdjklfa834fs()
    comp.finalize()
    with pytest.raises(TypeError) as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'jausasdjklfa834fs')

    def error_callback():
        raise ValueError("AAABBBCCC")

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.set('artiq/ttl0', True)
    s.add_step(0.01) \
      .pulse('artiq/urukul0_ch2/freq', test_utils.new_extern(lambda: 1.23)) \
      .pulse('artiq/ttl2', test_utils.new_extern(lambda: True))
    def ajsdf78ah4has9d():
        s.set('artiq/ttl0', test_utils.new_extern(error_callback))
    ajsdf78ah4has9d()
    comp.finalize()
    with pytest.raises(ValueError, match="AAABBBCCC") as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'ajsdf78ah4has9d')

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.set('artiq/ttl0', True)
    s.add_step(0.01) \
      .pulse('artiq/urukul0_ch2/freq', test_utils.new_extern(lambda: 1.23)) \
      .pulse('artiq/ttl2', test_utils.new_extern(lambda: True))
    def jas830bnsod8q():
        s.set('artiq/urukul3_ch0/amp', test_utils.new_extern(error_callback))
    jas830bnsod8q()
    comp.finalize()
    with pytest.raises(ValueError, match="AAABBBCCC") as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'jas830bnsod8q')

@with_artiq_params
def test_channels(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    assert s.get_channel_id('artiq/ttl0') == 0
    assert s.get_channel_id('artiq/ttl1') == 1
    assert s.get_channel_id('artiq/ttl10_counter') == 2
    assert s.get_channel_id('rfsoc/ch1/0/freq') == 3
    assert s.get_channel_id('artiq/urukul3_ch0/freq') == 4
    assert s.get_channel_id('artiq/urukul3_ch0/amp') == 5
    assert s.get_channel_id('artiq/urukul3_ch1/phase') == 6
    assert s.get_channel_id('artiq/urukul3_ch2/sw') == 7
    assert s.get_channel_id('artiq/urukul0_ch0/sw') == 8
    comp.finalize()
    channels = comp.get_channel_info()
    assert channels.ttl_chn_map == {0: 0, 1: 1, 2: 2, 7: 3, 8: 4}
    assert not channels.ttlchns[0].iscounter
    assert not channels.ttlchns[1].iscounter
    assert channels.ttlchns[2].iscounter
    assert channels.dds_param_chn_map == {4: (0, 'ddsfreq'), 5: (0, 'ddsamp'), 6: (1, 'ddsphase')}
    compiled_info = comp.get_compiled_info()
    assert not compiled_info.all_actions[0]
    assert not compiled_info.bool_values
    assert not compiled_info.float_values
    assert not compiled_info.relocations

    comp = test_env.new_comp(max_bt, dummy_artiq.HasEnvironment())
    s = comp.seq
    assert s.get_channel_id('artiq/ttl0') == 0
    assert s.get_channel_id('artiq/urukul3_ch0/freq') == 1
    assert s.get_channel_id('artiq/ttl1') == 2
    assert s.get_channel_id('artiq/urukul3_ch0/amp') == 3
    assert s.get_channel_id('artiq/ttl10_counter') == 4
    assert s.get_channel_id('artiq/urukul3_ch1/phase') == 5
    assert s.get_channel_id('rfsoc/ch1/0/freq') == 6
    assert s.get_channel_id('artiq/urukul3_ch2/sw') == 7
    assert s.get_channel_id('artiq/urukul0_ch0/sw') == 8
    comp.finalize()
    channels = comp.get_channel_info()
    assert channels.ttl_chn_map == {0: 0, 2: 1, 4: 2, 7: 3, 8: 4}
    assert not channels.ttlchns[0].iscounter
    assert not channels.ttlchns[1].iscounter
    assert channels.ttlchns[2].iscounter
    assert channels.dds_param_chn_map == {1: (0, 'ddsfreq'), 3: (0, 'ddsamp'), 5: (1, 'ddsphase')}
    compiled_info = comp.get_compiled_info()
    assert not compiled_info.all_actions[0]
    assert not compiled_info.bool_values
    assert not compiled_info.float_values
    assert not compiled_info.relocations

    comp = test_env.new_comp(max_bt, dummy_artiq.HasEnvironment())
    s = comp.seq
    s.get_channel_id('artiq/urukul0_ch0/a/b')
    with pytest.raises(ValueError, match="Invalid channel name artiq/urukul0_ch0/a/b"):
        comp.finalize()

    comp = test_env.new_comp(max_bt, dummy_artiq.HasEnvironment())
    s = comp.seq
    s.get_channel_id('artiq/urukul0_ch0/aaa')
    with pytest.raises(ValueError, match="Invalid channel name artiq/urukul0_ch0/aaa"):
        comp.finalize()

    comp = test_env.new_comp(max_bt, dummy_artiq.HasEnvironment())
    s = comp.seq
    s.get_channel_id('artiq/ttl0/a')
    with pytest.raises(ValueError, match="Invalid channel name artiq/ttl0/a"):
        comp.finalize()

    comp = test_env.new_comp(max_bt, dummy_artiq.HasEnvironment())
    s = comp.seq
    s.get_channel_id('artiq/ttl0_counter/a')
    with pytest.raises(ValueError, match="Invalid channel name artiq/ttl0_counter/a"):
        comp.finalize()

    comp = test_env.new_comp(max_bt, dummy_artiq.HasEnvironment())
    s = comp.seq
    s.get_channel_id('artiq/something')
    with pytest.raises(Exception):
        comp.finalize()

    comp = test_env.new_comp(max_bt, dummy_artiq.HasEnvironment())
    s = comp.seq
    s.get_channel_id('artiq')
    with pytest.raises(ValueError, match="Invalid channel name artiq"):
        comp.finalize()

    comp = test_env.new_comp(max_bt, dummy_artiq.HasEnvironment())
    s = comp.seq
    s.get_channel_id('artiq/dummy_dev')
    with pytest.raises(ValueError, match="Unsupported device: "):
        comp.finalize()

@with_artiq_params
def test_action_kws(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    def ashdfjasldfas2a7sdfds(step):
        step.set('artiq/ttl0', True, arg=123)
    ashdfjasldfas2a7sdfds(s.add_step(0.01))
    with pytest.raises(ValueError, match="Invalid output keyword argument") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'ashdfjasldfas2a7sdfds')

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    def asdf89hsdf():
        s.add_step(0.01) \
         .set('artiq/ttl0_counter', True, b=True)
    asdf89hsdf()
    with pytest.raises(ValueError, match="Invalid output keyword argument") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'asdf89hsdf')

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    def ahsd87hfasdf():
        s.add_step(0.01) \
         .set('artiq/urukul0_ch1/freq', 1.2e7, b=True)
    ahsd87hfasdf()
    with pytest.raises(ValueError, match="Invalid output keyword argument") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'ahsd87hfasdf')

@with_artiq_params
def test_ttl(max_bt):
    v = test_utils.new_extern(lambda: True)
    v2 = test_utils.new_extern(lambda: 0.2)
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    @s.add_step
    def step(s):
        s.add_step(0.01) \
         .set('artiq/ttl0', True) \
         .pulse('artiq/ttl1', v, exact_time=True)
        s.wait(v2)
        s.conditional(v).add_step(0.01) \
         .set('artiq/ttl0', False, exact_time=False) \
         .pulse('artiq/ttl1', v)
        s.wait(0.001)
        s.add_step(0.01) \
         .set('artiq/ttl0', False, exact_time=True) \
         .set('artiq/ttl1', v)
    s.conditional(False).add_step(0.1).set('artiq/ttl3', True)
    comp.finalize()
    compiled_info = comp.get_compiled_info()
    assert len(compiled_info.all_actions[0]) == 8

    comp.runtime_finalize(1)
    channels = comp.get_channel_info()
    ttlchns = channels.ttlchns
    test_env.check_list(comp, [
        -3000, # wait 3000 ns
        ttlchns[1].target, 1, # TTL output, exact_time first
        ttlchns[0].target, 1,
        -10_000_000,
        ttlchns[1].target, 0, # Pulse end
        -200_000_000,
        ttlchns[0].target, 0, # Second action
        ttlchns[1].target, 1,
        -10_000_000,
        ttlchns[1].target, 0, # Pulse end
        -1_000_000,
        ttlchns[1].target, 1, # Third action, no-op set on ttl0 omitted
        -10_000_000,
    ])

    comp.runtime_finalize(2)
    test_env.check_unchanged(comp)

@with_artiq_params
def test_counter(max_bt):
    v = test_utils.new_extern(lambda: True)
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    @s.add_step
    def step(s):
        s.add_step(0.01) \
         .set('artiq/ttl0_counter', True) \
         .pulse('artiq/ttl1_counter', True, exact_time=True)
        s.wait(0.001)
        c = s.conditional(v)
        @c.add_step
        def step2(s):
            s.conditional(v).add_step(0.01) \
             .set('artiq/ttl0_counter', False, exact_time=False) \
             .pulse('artiq/ttl1_counter', True)
            s.wait(0.001)
            s.conditional(v).add_step(0.01) \
             .set('artiq/ttl0_counter', False, exact_time=False) \
             .pulse('artiq/ttl1_counter', True)
        s.wait(0.001)
        s.conditional(False).add_step(0.01) \
         .set('artiq/ttl0_counter', False) \
         .set('artiq/ttl1_counter', True)
    comp.finalize()
    compiled_info = comp.get_compiled_info()
    assert len(compiled_info.all_actions[0]) == 9

    comp.runtime_finalize(1)
    channels = comp.get_channel_info()
    ttlchns = channels.ttlchns
    test_env.check_list(comp, [
        -3000, # wait 3000 ns
        ttlchns[0].target, 9, # TTL output, action order
        ttlchns[1].target, 9,
        -10_000_000,
        ttlchns[1].target, 4, # Pulse end
        -1_000_000,
        ttlchns[0].target, 4, # Second action
        ttlchns[1].target, 9,
        -10_000_000,
        ttlchns[1].target, 4, # Pulse end
        -1_000_000,
        ttlchns[1].target, 9, # Third action
        -10_000_000,
        ttlchns[1].target, 4, # Pulse end
        -1_000_000,
    ])

    comp.runtime_finalize(2)
    test_env.check_unchanged(comp)

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    def ahsd87fasdfasf():
        s.add_step(0.01) \
         .set('artiq/ttl0_counter', v)
    ahsd87fasdfasf()
    with pytest.raises(ValueError, match="Counter value must be static") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'ahsd87fasdfasf')

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.conditional(v).add_step(0.01) \
     .set('artiq/ttl0_counter', True)
    s.wait(1e-6)
    def as78j3df89asdf():
        s.add_step(0.01) \
         .pulse('artiq/ttl0_counter', True)
    as78j3df89asdf()
    with pytest.raises(ValueError, match="Counter value must be static") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'as78j3df89asdf')

@with_artiq_params
def test_dds(max_bt):
    v = test_utils.new_extern(lambda: 123e6)
    v2 = test_utils.new_extern(lambda: 0.1)
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    @s.add_step
    def step(s):
        s.add_step(0.01) \
         .set('artiq/ttl0', True) \
         .pulse('artiq/urukul0_ch0/amp', v2) \
         .set('artiq/urukul0_ch0/phase', v2 - 1) \
         .set('artiq/urukul0_ch0/freq', v, exact_time=True)
        s.wait(0.001)
        s.add_step(0.01) \
         .set('artiq/ttl0', False, exact_time=False) \
         .pulse('artiq/urukul0_ch0/amp', 0.01) \
         .pulse('artiq/urukul0_ch1/amp', 1.2) \
         .set('artiq/urukul0_ch0/freq', 23e6)
        s.wait(0.001)
        s.add_step(0.01) \
         .pulse('artiq/urukul0_ch0/amp', -0.1) \
         .set('artiq/urukul0_ch0/phase', 1.2)
    s.conditional(False).add_step(0.1).set('artiq/urukul0_ch2/freq', 100e6)
    comp.finalize()
    compiled_info = comp.get_compiled_info()
    assert len(compiled_info.all_actions[0]) == 14

    channels = comp.get_channel_info()
    addr_tgts = [bus.addr_target for bus in channels.urukul_busses]
    data_tgts = [bus.data_target for bus in channels.urukul_busses]
    upd_tgts = [bus.io_update_target for bus in channels.urukul_busses]
    css = [dds.chip_select for dds in channels.ddschns]
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]

    comp.runtime_finalize(1)

    test_env.check_list(comp, [
        addr_tgts[0], dds_config_addr(css[0]), -dds_config_len,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0.1, 0.1), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(123e6), -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]), -(3000 - dds_total_len),
        # Beginning of sequence
        ttl_tgts[0], 1,
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0, 0.1), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(123e6), -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]), -(10_000_000 - dds_headless_len - 8),
        # End of first step.
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0.01, 0.1), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(23e6), -dds_data_len,
        addr_tgts[0], dds_config_addr(css[1]), -dds_config_len,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[1]), -dds_config_len,
        data_tgts[0], dds_data1(1.2, 0), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[1]), -dds_config_len,
        data_tgts[0], dds_data2(0), -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]),
        -(1_000_000 - dds_headless_len - dds_total_len - 8),
        # Beginning of second step
        ttl_tgts[0], 0,
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0, 0.1), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(23e6), -dds_data_len,
        addr_tgts[0], dds_config_addr(css[1]), -dds_config_len,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[1]), -dds_config_len,
        data_tgts[0], dds_data1(0, 0), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[1]), -dds_config_len,
        data_tgts[0], dds_data2(0), -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]),
        -(10_000_000 - dds_headless_len - dds_total_len - 8),
        # End of second step
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0, 1.2), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(23e6),
        -(1_000_000 - dds_headless_len + dds_data_len - 8),
        # Beginning of third step
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        -(10_000_000 - 8),
    ])

    comp.runtime_finalize(2)
    test_env.check_unchanged(comp)

@with_artiq_params
def test_ramp(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    def sdf67asdf7f8sdf():
        s.add_step(0.01) \
         .set('artiq/ttl0', test_utils.StaticFunction())
    sdf67asdf7f8sdf()
    with pytest.raises(ValueError, match="TTL Channel cannot be ramped") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'sdf67asdf7f8sdf')

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    def asdf787asd78hte78():
        s.add_step(0.01) \
         .set('artiq/ttl0_counter', test_utils.StaticFunction())
    asdf787asd78hte78()
    with pytest.raises(ValueError, match="TTL Channel cannot be ramped") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'asdf787asd78hte78')

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    def as8df9sdf8():
        s.add_step(0.01) \
         .set('artiq/urukul0_ch2/amp', test_utils.StaticFunction())
    as8df9sdf8()
    with pytest.raises(ValueError, match="DDS Channel cannot be ramped") as exc:
        comp.finalize()
    test_utils.check_bt(exc, max_bt, 'as8df9sdf8')

@with_artiq_params
def test_start_trigger(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.set('artiq/ttl10', True)
    artiq_utils.add_start_trigger(comp.ab, 0xfff1231, 0, 8000, True)
    artiq_utils.add_start_trigger(comp.ab, 0xfff1232, 0, 8000, False)
    comp.finalize()
    triggers = artiq_utils.get_start_trigger(comp.ab)
    assert len(triggers) == 2
    assert triggers[0].target == 0xfff1231
    assert triggers[0].min_time_mu == 8
    assert triggers[0].raising_edge
    assert triggers[0].time_mu == 0
    assert triggers[1].target == 0xfff1232
    assert triggers[1].min_time_mu == 8
    assert not triggers[1].raising_edge
    assert triggers[1].time_mu == 0

    comp.runtime_finalize(1)

    channels = comp.get_channel_info()
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]
    test_env.check_list(comp, [
        -3000,
        0xfff1232, 1, -8,
        0xfff1231, 1,
        0xfff1232, 0,
        ttl_tgts[0], 1, -8,
        0xfff1231, 0,
    ])

    comp.runtime_finalize(2)
    test_env.check_unchanged(comp)

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.set('artiq/ttl10', True)
    comp.ab.add_start_trigger('ttl11', 0, 8e-9, True)
    comp.ab.add_start_trigger('ttl12', 0, 8e-9, False)
    comp.finalize()
    triggers = artiq_utils.get_start_trigger(comp.ab)
    assert len(triggers) == 2
    trigger1 = triggers[0].target
    assert triggers[0].min_time_mu == 8
    assert triggers[0].raising_edge
    assert triggers[0].time_mu == 0
    trigger2 = triggers[1].target
    assert triggers[1].min_time_mu == 8
    assert not triggers[1].raising_edge
    assert triggers[1].time_mu == 0

    comp.runtime_finalize(1)

    channels = comp.get_channel_info()
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]
    test_env.check_list(comp, [
        -3000,
        trigger2, 1, -8,
        trigger1, 1,
        trigger2, 0,
        ttl_tgts[0], 1, -8,
        trigger1, 0,
    ])

    comp.runtime_finalize(2)
    test_env.check_unchanged(comp)

@with_artiq_params
def test_start_trigger_error(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    with pytest.raises(ValueError, match="Invalid start trigger device: ttl0_counter"):
        comp.ab.add_start_trigger('ttl0_counter', 0, 8e-9, True)
    with pytest.raises(ValueError, match="Invalid start trigger device: urukul1_ch0"):
        comp.ab.add_start_trigger('urukul1_ch0', 0, 8e-9, True)

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.set('artiq/ttl10', True)
    for i in range(8):
        artiq_utils.add_start_trigger(comp.ab, 0xfff1231 + i, 0, 8000, True)
    comp.finalize()

    with pytest.raises(ValueError, match="Too many start triggers at the same time"):
        comp.runtime_finalize(1)

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.set('artiq/ttl10', True)
    for t in range(200):
        for i in range(7):
            artiq_utils.add_start_trigger(comp.ab, 0xfff1231 + t * 8 + i, t * 8000,
                                          8000, True)
    comp.finalize()

    with pytest.raises(ValueError, match="Too many start triggers at the same time"):
        comp.runtime_finalize(1)

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.set('artiq/ttl10', True)
    for t in range(200):
        for i in range(7):
            artiq_utils.add_start_trigger(comp.ab, 0xfff1231 + t * 8 + i,
                                          t * 8000, 8000, False)
    comp.finalize()

    with pytest.raises(ValueError, match="Too many start triggers at the same time"):
        comp.runtime_finalize(1)

@with_artiq_params
def test_ttl_time_move(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    for i in range(8):
        s.set(f'artiq/ttl{i}', True)
    comp.finalize()

    channels = comp.get_channel_info()
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]

    comp.runtime_finalize(1)
    test_env.check_list(comp, [
        -2992,
        ttl_tgts[7], 1,
        -8,
        ttl_tgts[0], 1,
        ttl_tgts[1], 1,
        ttl_tgts[2], 1,
        ttl_tgts[3], 1,
        ttl_tgts[4], 1,
        ttl_tgts[5], 1,
        ttl_tgts[6], 1,
    ])

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    for i in range(15):
        s.set(f'artiq/ttl{i}', True)
    comp.finalize()

    channels = comp.get_channel_info()
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]

    comp.runtime_finalize(1)
    test_env.check_list(comp, [
        -2992,
        ttl_tgts[7], 1,
        ttl_tgts[8], 1,
        ttl_tgts[9], 1,
        ttl_tgts[10], 1,
        ttl_tgts[11], 1,
        ttl_tgts[12], 1,
        ttl_tgts[13], 1,
        -8,
        ttl_tgts[0], 1,
        ttl_tgts[1], 1,
        ttl_tgts[2], 1,
        ttl_tgts[3], 1,
        ttl_tgts[4], 1,
        ttl_tgts[5], 1,
        ttl_tgts[6], 1,
        -8,
        ttl_tgts[14], 1,
    ])

@with_artiq_params
def test_ttl_exact_time(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.set(f'artiq/ttl0', True, exact_time=True)
    s.wait(8e-9)
    s.set(f'artiq/ttl0', False, exact_time=True)
    comp.finalize()

    channels = comp.get_channel_info()
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]

    comp.runtime_finalize(1)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 1,
        -8,
        ttl_tgts[0], 0,
    ])

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.set(f'artiq/ttl0', True)
    s.wait(8e-9)
    s.set(f'artiq/ttl0', False, exact_time=True)
    comp.finalize()

    channels = comp.get_channel_info()
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]

    comp.runtime_finalize(1)
    test_env.check_list(comp, [
        -3008,
        ttl_tgts[0], 0,
    ])

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    for i in range(7):
        s.set(f'artiq/ttl{i}', True, exact_time=True)
    comp.finalize()

    channels = comp.get_channel_info()
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]

    comp.runtime_finalize(1)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 1,
        ttl_tgts[1], 1,
        ttl_tgts[2], 1,
        ttl_tgts[3], 1,
        ttl_tgts[4], 1,
        ttl_tgts[5], 1,
        ttl_tgts[6], 1,
    ])

@with_artiq_params
def test_dds_merge_value(max_bt):
    v1 = 1e-6
    rv1 = test_utils.new_extern(lambda: v1)
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.set(f'artiq/urukul0_ch0/amp', 0.1)
    s.set(f'artiq/urukul0_ch0/freq', 1.3e6)
    s.wait(0.1e-3)
    s.set(f'artiq/urukul0_ch0/amp', 0.2)
    s.wait(rv1)
    s.set(f'artiq/urukul0_ch0/amp', 0.1)
    comp.finalize()

    channels = comp.get_channel_info()
    addr_tgts = [bus.addr_target for bus in channels.urukul_busses]
    data_tgts = [bus.data_target for bus in channels.urukul_busses]
    upd_tgts = [bus.io_update_target for bus in channels.urukul_busses]
    css = [dds.chip_select for dds in channels.ddschns]
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]

    comp.runtime_finalize(1)
    test_env.check_list(comp, [
        addr_tgts[0], dds_config_addr(css[0]),
        -dds_config_len,
        data_tgts[0], dds_data_addr,
        -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]),
        -dds_config_len,
        data_tgts[0], dds_data1(0.1, 0),
        -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]),
        -dds_config_len,
        data_tgts[0], dds_data2(1.3e6),
        -(3000 - dds_total_len + dds_data_len),
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        -(101_000 - 8),
    ])

    v1 = 5e-6
    comp.runtime_finalize(2)
    test_env.check_list(comp, [
        addr_tgts[0], dds_config_addr(css[0]),
        -dds_config_len,
        data_tgts[0], dds_data_addr,
        -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]),
        -dds_config_len,
        data_tgts[0], dds_data1(0.1, 0),
        -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]),
        -dds_config_len,
        data_tgts[0], dds_data2(1.3e6),
        -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]),
        -(3000 - dds_total_len),
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr,
        -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]),
        -dds_config_len,
        data_tgts[0], dds_data1(0.2, 0),
        -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]),
        -dds_config_len,
        data_tgts[0], dds_data2(1.3e6),
        -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]),
        -(100_000 - 8 - dds_total_len + dds_config_len),
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr,
        -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]),
        -dds_config_len,
        data_tgts[0], dds_data1(0.1, 0),
        -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]),
        -dds_config_len,
        data_tgts[0], dds_data2(1.3e6),
        -(5_000 - 8 - dds_headless_len + dds_data_len),
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
    ])

@with_artiq_params
def test_dds_exact_time(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    for i in range(7):
        s.set(f'artiq/urukul{i}_ch{i % 4}/amp', 0.1, exact_time=True)
        s.set(f'artiq/urukul{i}_ch{i % 4}/freq', (12 + i) * 1e6, exact_time=True)
    comp.finalize()

    channels = comp.get_channel_info()
    addr_tgts = [bus.addr_target for bus in channels.urukul_busses]
    data_tgts = [bus.data_target for bus in channels.urukul_busses]
    upd_tgts = [bus.io_update_target for bus in channels.urukul_busses]
    css = [dds.chip_select for dds in channels.ddschns]
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]

    comp.runtime_finalize(1)
    test_env.check_list(comp, [
        addr_tgts[0], dds_config_addr(css[0]),
        addr_tgts[1], dds_config_addr(css[1]),
        addr_tgts[2], dds_config_addr(css[2]),
        addr_tgts[3], dds_config_addr(css[3]),
        addr_tgts[4], dds_config_addr(css[4]),
        addr_tgts[5], dds_config_addr(css[5]),
        addr_tgts[6], dds_config_addr(css[6]),
        -dds_config_len,
        data_tgts[0], dds_data_addr,
        data_tgts[1], dds_data_addr,
        data_tgts[2], dds_data_addr,
        data_tgts[3], dds_data_addr,
        data_tgts[4], dds_data_addr,
        data_tgts[5], dds_data_addr,
        data_tgts[6], dds_data_addr,
        -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]),
        addr_tgts[1], dds_config_data1(css[1]),
        addr_tgts[2], dds_config_data1(css[2]),
        addr_tgts[3], dds_config_data1(css[3]),
        addr_tgts[4], dds_config_data1(css[4]),
        addr_tgts[5], dds_config_data1(css[5]),
        addr_tgts[6], dds_config_data1(css[6]),
        -dds_config_len,
        data_tgts[0], dds_data1(0.1, 0),
        data_tgts[1], dds_data1(0.1, 0),
        data_tgts[2], dds_data1(0.1, 0),
        data_tgts[3], dds_data1(0.1, 0),
        data_tgts[4], dds_data1(0.1, 0),
        data_tgts[5], dds_data1(0.1, 0),
        data_tgts[6], dds_data1(0.1, 0),
        -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]),
        addr_tgts[1], dds_config_data2(css[1]),
        addr_tgts[2], dds_config_data2(css[2]),
        addr_tgts[3], dds_config_data2(css[3]),
        addr_tgts[4], dds_config_data2(css[4]),
        addr_tgts[5], dds_config_data2(css[5]),
        addr_tgts[6], dds_config_data2(css[6]),
        -dds_config_len,
        data_tgts[0], dds_data2(12e6),
        data_tgts[1], dds_data2(13e6),
        data_tgts[2], dds_data2(14e6),
        data_tgts[3], dds_data2(15e6),
        data_tgts[4], dds_data2(16e6),
        data_tgts[5], dds_data2(17e6),
        data_tgts[6], dds_data2(18e6),
        -(3000 - dds_total_len + dds_data_len),
        upd_tgts[0], 1,
        upd_tgts[1], 1,
        upd_tgts[2], 1,
        upd_tgts[3], 1,
        upd_tgts[4], 1,
        upd_tgts[5], 1,
        upd_tgts[6], 1,
        -8,
        upd_tgts[0], 0,
        upd_tgts[1], 0,
        upd_tgts[2], 0,
        upd_tgts[3], 0,
        upd_tgts[4], 0,
        upd_tgts[5], 0,
        upd_tgts[6], 0,
    ])

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    for i in range(7):
        s.set(f'artiq/urukul{i}_ch{i % 4}/amp', 0.1, exact_time=True)
        s.set(f'artiq/urukul{i}_ch{i % 4}/freq', (12 + i) * 1e6, exact_time=True)
    s.wait(1.5e-6)
    for i in range(7):
        s.set(f'artiq/urukul{i}_ch{i % 4}/amp', 0.2, exact_time=True)
    comp.finalize()

    channels = comp.get_channel_info()
    addr_tgts = [bus.addr_target for bus in channels.urukul_busses]
    data_tgts = [bus.data_target for bus in channels.urukul_busses]
    upd_tgts = [bus.io_update_target for bus in channels.urukul_busses]
    css = [dds.chip_select for dds in channels.ddschns]
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]

    comp.runtime_finalize(1)
    test_env.check_list(comp, [
        addr_tgts[0], dds_config_addr(css[0]),
        addr_tgts[1], dds_config_addr(css[1]),
        addr_tgts[2], dds_config_addr(css[2]),
        addr_tgts[3], dds_config_addr(css[3]),
        addr_tgts[4], dds_config_addr(css[4]),
        addr_tgts[5], dds_config_addr(css[5]),
        addr_tgts[6], dds_config_addr(css[6]),
        -dds_config_len,
        data_tgts[0], dds_data_addr,
        data_tgts[1], dds_data_addr,
        data_tgts[2], dds_data_addr,
        data_tgts[3], dds_data_addr,
        data_tgts[4], dds_data_addr,
        data_tgts[5], dds_data_addr,
        data_tgts[6], dds_data_addr,
        -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]),
        addr_tgts[1], dds_config_data1(css[1]),
        addr_tgts[2], dds_config_data1(css[2]),
        addr_tgts[3], dds_config_data1(css[3]),
        addr_tgts[4], dds_config_data1(css[4]),
        addr_tgts[5], dds_config_data1(css[5]),
        addr_tgts[6], dds_config_data1(css[6]),
        -dds_config_len,
        data_tgts[0], dds_data1(0.1, 0),
        data_tgts[1], dds_data1(0.1, 0),
        data_tgts[2], dds_data1(0.1, 0),
        data_tgts[3], dds_data1(0.1, 0),
        data_tgts[4], dds_data1(0.1, 0),
        data_tgts[5], dds_data1(0.1, 0),
        data_tgts[6], dds_data1(0.1, 0),
        -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]),
        addr_tgts[1], dds_config_data2(css[1]),
        addr_tgts[2], dds_config_data2(css[2]),
        addr_tgts[3], dds_config_data2(css[3]),
        addr_tgts[4], dds_config_data2(css[4]),
        addr_tgts[5], dds_config_data2(css[5]),
        addr_tgts[6], dds_config_data2(css[6]),
        -dds_config_len,
        data_tgts[0], dds_data2(12e6),
        data_tgts[1], dds_data2(13e6),
        data_tgts[2], dds_data2(14e6),
        data_tgts[3], dds_data2(15e6),
        data_tgts[4], dds_data2(16e6),
        data_tgts[5], dds_data2(17e6),
        data_tgts[6], dds_data2(18e6),
        -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]),
        addr_tgts[1], dds_config_addr(css[1]),
        addr_tgts[2], dds_config_addr(css[2]),
        addr_tgts[3], dds_config_addr(css[3]),
        addr_tgts[4], dds_config_addr(css[4]),
        addr_tgts[5], dds_config_addr(css[5]),
        addr_tgts[6], dds_config_addr(css[6]),
        -(3000 - dds_total_len),
        upd_tgts[0], 1,
        upd_tgts[1], 1,
        upd_tgts[2], 1,
        upd_tgts[3], 1,
        upd_tgts[4], 1,
        upd_tgts[5], 1,
        upd_tgts[6], 1,
        -8,
        upd_tgts[0], 0,
        upd_tgts[1], 0,
        upd_tgts[2], 0,
        upd_tgts[3], 0,
        upd_tgts[4], 0,
        upd_tgts[5], 0,
        upd_tgts[6], 0,
        -8,
        data_tgts[0], dds_data_addr,
        data_tgts[1], dds_data_addr,
        data_tgts[2], dds_data_addr,
        data_tgts[3], dds_data_addr,
        data_tgts[4], dds_data_addr,
        data_tgts[5], dds_data_addr,
        data_tgts[6], dds_data_addr,
        -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]),
        addr_tgts[1], dds_config_data1(css[1]),
        addr_tgts[2], dds_config_data1(css[2]),
        addr_tgts[3], dds_config_data1(css[3]),
        addr_tgts[4], dds_config_data1(css[4]),
        addr_tgts[5], dds_config_data1(css[5]),
        addr_tgts[6], dds_config_data1(css[6]),
        -dds_config_len,
        data_tgts[0], dds_data1(0.2, 0),
        data_tgts[1], dds_data1(0.2, 0),
        data_tgts[2], dds_data1(0.2, 0),
        data_tgts[3], dds_data1(0.2, 0),
        data_tgts[4], dds_data1(0.2, 0),
        data_tgts[5], dds_data1(0.2, 0),
        data_tgts[6], dds_data1(0.2, 0),
        -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]),
        addr_tgts[1], dds_config_data2(css[1]),
        addr_tgts[2], dds_config_data2(css[2]),
        addr_tgts[3], dds_config_data2(css[3]),
        addr_tgts[4], dds_config_data2(css[4]),
        addr_tgts[5], dds_config_data2(css[5]),
        addr_tgts[6], dds_config_data2(css[6]),
        -dds_config_len,
        data_tgts[0], dds_data2(12e6),
        data_tgts[1], dds_data2(13e6),
        data_tgts[2], dds_data2(14e6),
        data_tgts[3], dds_data2(15e6),
        data_tgts[4], dds_data2(16e6),
        data_tgts[5], dds_data2(17e6),
        data_tgts[6], dds_data2(18e6),
        # 1.5 us is not 8 ns aligned so it's 1504 here
        -(1504 - dds_headless_len - 16 + dds_data_len),
        upd_tgts[0], 1,
        upd_tgts[1], 1,
        upd_tgts[2], 1,
        upd_tgts[3], 1,
        upd_tgts[4], 1,
        upd_tgts[5], 1,
        upd_tgts[6], 1,
        -8,
        upd_tgts[0], 0,
        upd_tgts[1], 0,
        upd_tgts[2], 0,
        upd_tgts[3], 0,
        upd_tgts[4], 0,
        upd_tgts[5], 0,
        upd_tgts[6], 0,
    ])

@with_artiq_params
def test_exact_time_error(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    for i in range(7):
        s.set(f'artiq/ttl{i}', True, exact_time=True)
    def adsf78as7dfahsd78f():
        s.set('artiq/ttl7', True, exact_time=True)
    adsf78as7dfahsd78f()
    comp.finalize()

    with pytest.raises(ValueError, match="Too many outputs at the same time") as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'adsf78as7dfahsd78f')

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    for i in range(7):
        s.set(f'artiq/urukul{i}_ch2/amp', True, exact_time=True)
        s.set(f'artiq/urukul{i}_ch2/freq', (12 + i) * 1e6, exact_time=True)
    def asdh78asdfasdj():
        s.set('artiq/urukul7_ch2/amp', True, exact_time=True)
        s.set('artiq/urukul7_ch2/freq', 20e6, exact_time=True)
    asdh78asdfasdj()
    comp.finalize()

    with pytest.raises(ValueError, match="Too many outputs at the same time") as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'asdh78asdfasdj')

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.set(f'artiq/urukul0_ch0/amp', True, exact_time=True)
    s.wait(0.1e-6)
    def ajsdj7jf8asdf():
        s.set(f'artiq/urukul0_ch1/amp', True, exact_time=True)
    ajsdj7jf8asdf()
    comp.finalize()

    with pytest.raises(ValueError, match="Exact time output cannot satisfy lower time bound") as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'ajsdj7jf8asdf')

@with_artiq_params
def test_inexact_time_error(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    def jasdjf78asdfhsd():
        for t in range(2000):
            step = s.add_step(8e-9)
            for i in range(16):
                step.set(f'artiq/ttl{i}_counter', t % 2 == 0)
    jasdjf78asdfhsd()
    comp.finalize()

    with pytest.raises(ValueError, match="Too many outputs at the same time") as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'jasdjf78asdfhsd')

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    def asd7923j9fd7():
        for t in range(100):
            step = s.add_step(0.5e-6)
            for i in range(4):
                step.set(f'artiq/urukul0_ch{i}/amp', t * 0.001)
    asd7923j9fd7()
    comp.finalize()

    with pytest.raises(ValueError, match="Cannot find appropriate output time within bound") as exc:
        comp.runtime_finalize(1)
    test_utils.check_bt(exc, max_bt, 'asd7923j9fd7')

@with_artiq_params
def test_long_wait(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.wait(10)

    comp.finalize()

    comp.runtime_finalize(1)
    test_env.check_list(comp, [
        -2147483648,
        -2147483648,
        -2147483648,
        -2147483648,
        -1410068408,
    ])

@with_artiq_params
def test_dyn_seq1(max_bt):
    c1 = True
    rc1 = test_utils.new_extern(lambda: c1)
    c2 = True
    rc2 = test_utils.new_extern(lambda: c2)

    v1 = 0.1
    rv1 = test_utils.new_extern(lambda: v1)
    v2 = 0.2
    rv2 = test_utils.new_extern(lambda: v2)

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.add_step(rv1) \
     .pulse('artiq/ttl0', rc2)
    s.conditional(rc1) \
     .add_step(0.001) \
     .set('artiq/ttl1', rc2)
    s.wait(rv2)
    s.add_step(rv1) \
     .pulse('artiq/ttl0', True)
    comp.finalize()

    channels = comp.get_channel_info()
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]

    comp.runtime_finalize(1)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 1,
        -100_000_000,
        ttl_tgts[0], 0,
        ttl_tgts[1], 1,
        -201_000_000,
        ttl_tgts[0], 1,
        -100_000_000,
        ttl_tgts[0], 0,
    ])

    c1 = False
    c2 = True
    v1 = 0.1
    v2 = 0.2
    comp.runtime_finalize(2)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 1,
        -100_000_000,
        ttl_tgts[0], 0,
        -200_000_000,
        ttl_tgts[0], 1,
        -100_000_000,
        ttl_tgts[0], 0,
    ])

    c1 = True
    c2 = False
    v1 = 0.1
    v2 = 0.2
    comp.runtime_finalize(3)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 0,
        -100_000_000,
        ttl_tgts[1], 0,
        -201_000_000,
        ttl_tgts[0], 1,
        -100_000_000,
        ttl_tgts[0], 0,
    ])

    c1 = False
    c2 = False
    v1 = 0.1
    v2 = 0.2
    comp.runtime_finalize(4)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 0,
        -300_000_000,
        ttl_tgts[0], 1,
        -100_000_000,
        ttl_tgts[0], 0,
    ])

    c1 = True
    c2 = True
    v1 = 0.0
    v2 = 0.2
    comp.runtime_finalize(5)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 0,
        ttl_tgts[1], 1,
        -201_000_000,
    ])

    c1 = False
    c2 = True
    v1 = 0.0
    v2 = 0.2
    comp.runtime_finalize(6)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 0,
        -200_000_000,
    ])

    c1 = True
    c2 = False
    v1 = 0.0
    v2 = 0.2
    comp.runtime_finalize(7)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 0,
        ttl_tgts[1], 0,
        -201_000_000,
    ])

    c1 = False
    c2 = False
    v1 = 0.0
    v2 = 0.2
    comp.runtime_finalize(8)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 0,
        -200_000_000,
    ])

    c1 = True
    c2 = True
    v1 = 0.1
    v2 = 0.0
    comp.runtime_finalize(9)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 1,
        -100_000_000,
        ttl_tgts[0], 0,
        ttl_tgts[1], 1,
        -1_000_000,
        ttl_tgts[0], 1,
        -100_000_000,
        ttl_tgts[0], 0,
    ])

    c1 = False
    c2 = True
    v1 = 0.1
    v2 = 0.0
    comp.runtime_finalize(10)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 1,
        -200_000_000,
        ttl_tgts[0], 0,
    ])

    c1 = True
    c2 = False
    v1 = 0.1
    v2 = 0.0
    comp.runtime_finalize(11)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 0,
        -100_000_000,
        ttl_tgts[1], 0,
        -1_000_000,
        ttl_tgts[0], 1,
        -100_000_000,
        ttl_tgts[0], 0,
    ])

    c1 = False
    c2 = False
    v1 = 0.1
    v2 = 0.0
    comp.runtime_finalize(12)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 0,
        -100_000_000,
        ttl_tgts[0], 1,
        -100_000_000,
        ttl_tgts[0], 0,
    ])

    c1 = True
    c2 = True
    v1 = 0.0
    v2 = 0.0
    comp.runtime_finalize(13)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 0,
        ttl_tgts[1], 1,
        -1_000_000,
    ])

    c1 = False
    c2 = True
    v1 = 0.0
    v2 = 0.0
    comp.runtime_finalize(14)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 0,
    ])

    c1 = True
    c2 = False
    v1 = 0.0
    v2 = 0.0
    comp.runtime_finalize(15)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 0,
        ttl_tgts[1], 0,
        -1_000_000,
    ])

    c1 = False
    c2 = False
    v1 = 0.0
    v2 = 0.0
    comp.runtime_finalize(16)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 0,
    ])

@with_artiq_params
def test_single(max_bt):
    b1 = True
    b2 = True
    v1 = 0.001
    rb1 = test_utils.new_extern(lambda: b1)
    rb2 = test_utils.new_extern(lambda: b2)
    rv1 = test_utils.new_extern(lambda: v1)

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.wait(rv1)
    s.conditional(rb1).set('artiq/ttl0', rb2)
    comp.finalize()

    channels = comp.get_channel_info()
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]

    comp.runtime_finalize(1)
    test_env.check_list(comp, [
        -1_003_000,
        ttl_tgts[0], 1,
    ])

    v1 = 0.001
    b1 = True
    b2 = False
    comp.runtime_finalize(2)
    test_env.check_list(comp, [
        -1_003_000,
        ttl_tgts[0], 0,
    ])

    v1 = 0.001
    b1 = False
    b2 = True
    comp.runtime_finalize(3)
    test_env.check_list(comp, [
        -1_003_000,
    ])

    v1 = 0.001
    b1 = False
    b2 = False
    comp.runtime_finalize(4)
    test_env.check_list(comp, [
        -1_003_000,
    ])

    v1 = 0.01
    b1 = True
    b2 = True
    comp.runtime_finalize(5)
    test_env.check_list(comp, [
        -10_003_000,
        ttl_tgts[0], 1,
    ])

    v1 = 0.01
    b1 = True
    b2 = False
    comp.runtime_finalize(6)
    test_env.check_list(comp, [
        -10_003_000,
        ttl_tgts[0], 0,
    ])

    v1 = 0.01
    b1 = False
    b2 = True
    comp.runtime_finalize(7)
    test_env.check_list(comp, [
        -10_003_000,
    ])

    v1 = 0.01
    b1 = False
    b2 = False
    comp.runtime_finalize(8)
    test_env.check_list(comp, [
        -10_003_000,
    ])

@with_artiq_params
def test_same_time_output(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    st1 = s.add_step(10e-6)
    st2 = s.add_step(10e-6)
    st1.pulse('artiq/ttl0', True)
    st2.pulse('artiq/ttl0', True)
    comp.finalize()

    channels = comp.get_channel_info()
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]

    comp.runtime_finalize(1)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 1,
        -20_000,
        ttl_tgts[0], 0,
    ])

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    st1 = s.add_step(10e-6)
    st2 = s.add_step(10e-6)
    st2.pulse('artiq/ttl0', True)
    st1.pulse('artiq/ttl0', True)
    comp.finalize()

    channels = comp.get_channel_info()
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]

    comp.runtime_finalize(1)
    test_env.check_list(comp, [
        -3000,
        ttl_tgts[0], 1,
        -20_000,
        ttl_tgts[0], 0,
    ])

@with_artiq_params
def test_device_delay_rt_error(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    comp.ab.set_device_delay("ttl0", test_utils.new_extern(lambda: -0.001))
    s.set('artiq/ttl0', True)
    comp.finalize()
    with pytest.raises(ValueError,
                       match="Device time offset -0.001 cannot be negative."):
        comp.runtime_finalize(1)

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    comp.ab.set_device_delay("urukul0_ch2", test_utils.new_extern(lambda: 1.2))
    s.set('artiq/urukul0_ch2/amp', True)
    comp.finalize()
    with pytest.raises(ValueError,
                       match="Device time offset 1.2 cannot be more than 100ms."):
        comp.runtime_finalize(1)

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    comp.ab.set_device_delay("ttl0", test_utils.new_extern(lambda: []))
    s.set('artiq/ttl0', True)
    comp.finalize()
    with pytest.raises(TypeError):
        comp.runtime_finalize(1)

    def error_callback():
        raise ValueError("AAABBBCCC")

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    comp.ab.set_device_delay("ttl0", test_utils.new_extern(error_callback))
    s.set('artiq/ttl0', True)
    comp.finalize()
    with pytest.raises(ValueError, match="AAABBBCCC"):
        comp.runtime_finalize(1)

@with_artiq_params
@pytest.mark.parametrize('use_rt', [False, True])
def test_device_delay(max_bt, use_rt):
    def wrap_value(v):
        if use_rt:
            return test_utils.new_extern(lambda: v)
        return v
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    with pytest.raises(ValueError,
                       match="Device time offset -0.001 cannot be negative."):
        comp.ab.set_device_delay("ttl0", -0.001)
    with pytest.raises(ValueError,
                       match="Device time offset 1.0 cannot be more than 100ms."):
        comp.ab.set_device_delay("urukul0_ch1", 1)

    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    comp.ab.set_device_delay("ttl1", wrap_value(1e-3))
    comp.ab.set_device_delay("ttl2_counter", wrap_value(2e-3))
    comp.ab.set_device_delay("urukul0_ch0", wrap_value(3e-3))

    s.add_step(test_utils.new_extern(lambda: 5e-3)) \
      .pulse('artiq/ttl1', True) \
      .pulse('artiq/ttl0', True) \
      .pulse('artiq/ttl2_counter', True) \
      .pulse('artiq/ttl3_counter', True) \
      .pulse('artiq/urukul0_ch0/sw', True) \
      .pulse('artiq/urukul1_ch0/sw', True) \
      .pulse('artiq/urukul0_ch0/amp', 0.2) \
      .pulse('artiq/urukul0_ch1/amp', 0.3)
    comp.finalize()

    channels = comp.get_channel_info()
    addr_tgts = [bus.addr_target for bus in channels.urukul_busses]
    data_tgts = [bus.data_target for bus in channels.urukul_busses]
    upd_tgts = [bus.io_update_target for bus in channels.urukul_busses]
    css = [dds.chip_select for dds in channels.ddschns]
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]

    comp.runtime_finalize(1)
    test_env.check_list(comp, [
        addr_tgts[0], dds_config_addr(css[1]),
        -dds_config_len,
        data_tgts[0], dds_data_addr,
        -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[1]),
        -dds_config_len,
        data_tgts[0], dds_data1(0.3, 0),
        -dds_data_len,
        addr_tgts[0], dds_config_data2(css[1]),
        -dds_config_len,
        data_tgts[0], dds_data2(0),
        -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]),
        -(3000 - dds_total_len),
        # Beginning of sequence
        ttl_tgts[3], 9,
        ttl_tgts[1], 1,
        ttl_tgts[5], 1,
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr,
        -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]),
        -dds_config_len,
        data_tgts[0], dds_data1(0.2, 0),
        -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]),
        -dds_config_len,
        data_tgts[0], dds_data2(0),
        -dds_data_len,
        addr_tgts[0], dds_config_addr(css[1]),
        -(1_000_000 - dds_headless_len - 8),
        # ttl0 delayed on @ 1 ms
        ttl_tgts[0], 1,
        -1_000_000,
        # ttl2_counter delayed on @ 2 ms
        ttl_tgts[2], 9,
        -1_000_000,
        # urukul0_ch0/sw and urukul0_ch0/amp delayed on @ 3 ms
        ttl_tgts[4], 1,
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr,
        -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[1]),
        -dds_config_len,
        data_tgts[0], dds_data1(0, 0),
        -dds_data_len,
        addr_tgts[0], dds_config_data2(css[1]),
        -dds_config_len,
        data_tgts[0], dds_data2(0),
        -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]),
        -(2_000_000 - dds_headless_len - 8),
        # end of step
        ttl_tgts[3], 4,
        ttl_tgts[1], 0,
        ttl_tgts[5], 0,
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr,
        -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]),
        -dds_config_len,
        data_tgts[0], dds_data1(0, 0),
        -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]),
        -dds_config_len,
        data_tgts[0], dds_data2(0),
        -(1_000_000 - dds_headless_len + dds_data_len - 8),
        # ttl0 delayed off @ 6 ms
        ttl_tgts[0], 0,
        -1_000_000,
        # ttl2_counter delayed off @ 7 ms
        ttl_tgts[2], 4,
        -1_000_000,
        # urukul0_ch0/sw and urukul0_ch0/amp delayed off @ 8 ms
        ttl_tgts[4], 0,
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
    ])

def test_empty_branch_backend():
    ab = artiq_backend.ArtiqBackend(dummy_artiq.DummyDaxSystem(), output_format="dma")
    assert ab.total_time_mu == 0

    ab = artiq_backend.ArtiqBackend(dummy_artiq.DummyDaxSystem(), output_format="bytecode")
    assert ab.total_time_mu == 0

@with_artiq_params
def test_branch_error(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    bs1 = s.new_basic_seq()
    s.add_branch(bs1)

    if not test_env.allow_branch:
        with pytest.raises(ValueError, match="not initialized for branch support"):
            comp.finalize()
        return
    comp.finalize()
    assert len(comp.ab.output) == 2
    assert comp.ab.output[0].bseq_id == 0
    assert not comp.ab.output[0].may_term
    assert comp.ab.output[0].next == [1]
    assert comp.ab.output[1].bseq_id == 1
    assert comp.ab.output[1].may_term
    assert comp.ab.output[1].next == []
    comp.runtime_finalize(1)
    # Builtin 3us wait
    test_env.check_lists(comp, [[-3000], [-3000]])
    assert comp.ab.output[0].total_time_mu == 3000
    assert comp.ab.output[1].total_time_mu == 3000

@with_artiq_br_params
def test_branch_ttl(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.add_step(0.1).set('artiq/ttl0', True)
    bs1 = s.new_basic_seq()
    s.add_branch(bs1)
    bs1.add_step(0.2).pulse('artiq/ttl0', False)

    comp.finalize()
    assert len(comp.ab.output) == 2
    assert comp.ab.output[0].bseq_id == 0
    assert not comp.ab.output[0].may_term
    assert comp.ab.output[0].next == [1]
    assert comp.ab.output[1].bseq_id == 1
    assert comp.ab.output[1].may_term
    assert comp.ab.output[1].next == []
    channels = comp.get_channel_info()
    ttlchns = channels.ttlchns

    comp.runtime_finalize(1)
    assert comp.ab.output[0].total_time_mu == 100_003_000
    assert comp.ab.output[1].total_time_mu == 200_003_000
    test_env.check_lists(comp, [[
        -3000,
        ttlchns[0].target, 1,
        -100_000_000
        ], [
        -3000,
        ttlchns[0].target, 0,
        -200_000_000,
        ttlchns[0].target, 1]])

    comp.runtime_finalize(2)
    test_env.check_unchanged(comp)

    b1 = True
    b2 = False
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.add_step(0.1).set('artiq/ttl0', test_utils.new_extern(lambda: b1))
    bs1 = s.new_basic_seq()
    s.add_branch(bs1)
    bs1.add_step(0.2).pulse('artiq/ttl0', test_utils.new_extern(lambda: b2))

    comp.finalize()
    assert len(comp.ab.output) == 2
    assert comp.ab.output[0].bseq_id == 0
    assert not comp.ab.output[0].may_term
    assert comp.ab.output[0].next == [1]
    assert comp.ab.output[1].bseq_id == 1
    assert comp.ab.output[1].may_term
    assert comp.ab.output[1].next == []
    channels = comp.get_channel_info()
    ttlchns = channels.ttlchns

    comp.runtime_finalize(1)
    assert comp.ab.output[0].total_time_mu == 100_003_000
    assert comp.ab.output[1].total_time_mu == 200_003_000
    test_env.check_lists(comp, [[
        -3000,
        ttlchns[0].target, 1,
        -100_000_000
        ], [
        -3000,
        ttlchns[0].target, 0,
        -200_000_000,
        ttlchns[0].target, 1]])

    comp.runtime_finalize(2)
    test_env.check_unchanged(comp)

    b1 = False
    b2 = False
    comp.runtime_finalize(3)
    assert comp.ab.output[0].total_time_mu == 100_003_000
    assert comp.ab.output[1].total_time_mu == 200_003_000
    test_env.check_lists(comp, [[
        -3000,
        ttlchns[0].target, 0,
        -100_000_000
        ], [
        -200_003_000]])

    comp.runtime_finalize(4)
    test_env.check_unchanged(comp)

    b1 = False
    b2 = True
    comp.runtime_finalize(5)
    assert comp.ab.output[0].total_time_mu == 100_003_000
    assert comp.ab.output[1].total_time_mu == 200_003_000
    test_env.check_lists(comp, [[
        -3000,
        ttlchns[0].target, 0,
        -100_000_000
        ], [
        -3000,
        ttlchns[0].target, 1,
        -200_000_000,
        ttlchns[0].target, 0]])

    comp.runtime_finalize(6)
    test_env.check_unchanged(comp)

    b1 = True
    b2 = True
    comp.runtime_finalize(7)
    assert comp.ab.output[0].total_time_mu == 100_003_000
    assert comp.ab.output[1].total_time_mu == 200_003_000
    test_env.check_lists(comp, [[
        -3000,
        ttlchns[0].target, 1,
        -100_000_000
        ], [
        -200_003_000]])

    comp.runtime_finalize(8)
    test_env.check_unchanged(comp)

@with_artiq_br_params
def test_branch_dds(max_bt):
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.add_step(0.1) \
         .pulse('artiq/urukul0_ch0/amp', 0.2) \
         .set('artiq/urukul0_ch0/phase', 0.7) \
         .set('artiq/urukul0_ch0/freq', 120e6)
    bs1 = s.new_basic_seq()
    s.add_branch(bs1)
    bs1.add_step(0.2).pulse('artiq/urukul0_ch0/amp', 0.4)

    comp.finalize()
    assert len(comp.ab.output) == 2
    assert comp.ab.output[0].bseq_id == 0
    assert not comp.ab.output[0].may_term
    assert comp.ab.output[0].next == [1]
    assert comp.ab.output[1].bseq_id == 1
    assert comp.ab.output[1].may_term
    assert comp.ab.output[1].next == []
    channels = comp.get_channel_info()
    addr_tgts = [bus.addr_target for bus in channels.urukul_busses]
    data_tgts = [bus.data_target for bus in channels.urukul_busses]
    upd_tgts = [bus.io_update_target for bus in channels.urukul_busses]
    css = [dds.chip_select for dds in channels.ddschns]
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]

    comp.runtime_finalize(1)
    assert comp.ab.output[0].total_time_mu == 100_003_008
    assert comp.ab.output[1].total_time_mu == 200_003_008
    test_env.check_lists(comp, [[
        addr_tgts[0], dds_config_addr(css[0]), -dds_config_len,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0.2, 0.7), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(120e6), -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]), -(3000 - dds_total_len),
        # Beginning of sequence
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0, 0.7), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(120e6),
        -(100_000_000 - dds_headless_len + dds_data_len - 8),
        # End of step
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        ], [
        addr_tgts[0], dds_config_addr(css[0]), -dds_config_len,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0.4, 0.7), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(120e6), -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]), -(3000 - dds_total_len),
        # Beginning of sequence
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0, 0.7), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(120e6),
        -(200_000_000 - dds_headless_len + dds_data_len - 8),
        # End of step
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0]])

    comp.runtime_finalize(2)

    amp1 = 0.2
    phase1 = 0.7
    freq1 = 120e6
    amp2 = 0.4
    test_env.check_unchanged(comp)
    comp = test_env.new_comp(max_bt, dummy_artiq.DummyDaxSystem())
    s = comp.seq
    s.add_step(0.1) \
         .pulse('artiq/urukul0_ch0/amp', test_utils.new_extern(lambda: amp1)) \
         .set('artiq/urukul0_ch0/phase', test_utils.new_extern(lambda: phase1)) \
         .set('artiq/urukul0_ch0/freq', test_utils.new_extern(lambda: freq1))
    bs1 = s.new_basic_seq()
    s.add_branch(bs1)
    bs1.add_step(0.2).pulse('artiq/urukul0_ch0/amp', test_utils.new_extern(lambda: amp2))

    comp.finalize()
    assert len(comp.ab.output) == 2
    assert comp.ab.output[0].bseq_id == 0
    assert not comp.ab.output[0].may_term
    assert comp.ab.output[0].next == [1]
    assert comp.ab.output[1].bseq_id == 1
    assert comp.ab.output[1].may_term
    assert comp.ab.output[1].next == []
    channels = comp.get_channel_info()
    addr_tgts = [bus.addr_target for bus in channels.urukul_busses]
    data_tgts = [bus.data_target for bus in channels.urukul_busses]
    upd_tgts = [bus.io_update_target for bus in channels.urukul_busses]
    css = [dds.chip_select for dds in channels.ddschns]
    ttl_tgts = [ttl.target for ttl in channels.ttlchns]

    comp.runtime_finalize(1)
    assert comp.ab.output[0].total_time_mu == 100_003_008
    assert comp.ab.output[1].total_time_mu == 200_003_008
    test_env.check_lists(comp, [[
        addr_tgts[0], dds_config_addr(css[0]), -dds_config_len,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0.2, 0.7), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(120e6), -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]), -(3000 - dds_total_len),
        # Beginning of sequence
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0, 0.7), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(120e6),
        -(100_000_000 - dds_headless_len + dds_data_len - 8),
        # End of step
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        ], [
        addr_tgts[0], dds_config_addr(css[0]), -dds_config_len,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0.4, 0.7), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(120e6), -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]), -(3000 - dds_total_len),
        # Beginning of sequence
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0, 0.7), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(120e6),
        -(200_000_000 - dds_headless_len + dds_data_len - 8),
        # End of step
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0]])

    comp.runtime_finalize(2)
    test_env.check_unchanged(comp)

    freq1 = 100e6
    comp.runtime_finalize(3)
    assert comp.ab.output[0].total_time_mu == 100_003_008
    assert comp.ab.output[1].total_time_mu == 200_003_008
    test_env.check_lists(comp, [[
        addr_tgts[0], dds_config_addr(css[0]), -dds_config_len,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0.2, 0.7), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(100e6), -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]), -(3000 - dds_total_len),
        # Beginning of sequence
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0, 0.7), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(100e6),
        -(100_000_000 - dds_headless_len + dds_data_len - 8),
        # End of step
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        ], [
        addr_tgts[0], dds_config_addr(css[0]), -dds_config_len,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0.4, 0.7), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(100e6), -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]), -(3000 - dds_total_len),
        # Beginning of sequence
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0, 0.7), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(100e6),
        -(200_000_000 - dds_headless_len + dds_data_len - 8),
        # End of step
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0]])

    comp.runtime_finalize(4)
    test_env.check_unchanged(comp)

    phase1 = 0.3
    comp.runtime_finalize(5)
    assert comp.ab.output[0].total_time_mu == 100_003_008
    assert comp.ab.output[1].total_time_mu == 200_003_008
    test_env.check_lists(comp, [[
        addr_tgts[0], dds_config_addr(css[0]), -dds_config_len,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0.2, 0.3), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(100e6), -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]), -(3000 - dds_total_len),
        # Beginning of sequence
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0, 0.3), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(100e6),
        -(100_000_000 - dds_headless_len + dds_data_len - 8),
        # End of step
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        ], [
        addr_tgts[0], dds_config_addr(css[0]), -dds_config_len,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0.4, 0.3), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(100e6), -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]), -(3000 - dds_total_len),
        # Beginning of sequence
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0, 0.3), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(100e6),
        -(200_000_000 - dds_headless_len + dds_data_len - 8),
        # End of step
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0]])

    comp.runtime_finalize(6)
    test_env.check_unchanged(comp)

    amp2 = 0
    comp.runtime_finalize(7)
    assert comp.ab.output[0].total_time_mu == 100_003_008
    assert comp.ab.output[1].total_time_mu == 200_003_000
    test_env.check_lists(comp, [[
        addr_tgts[0], dds_config_addr(css[0]), -dds_config_len,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0.2, 0.3), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(100e6), -dds_data_len,
        addr_tgts[0], dds_config_addr(css[0]), -(3000 - dds_total_len),
        # Beginning of sequence
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        data_tgts[0], dds_data_addr, -dds_addr_len,
        addr_tgts[0], dds_config_data1(css[0]), -dds_config_len,
        data_tgts[0], dds_data1(0, 0.3), -dds_data_len,
        addr_tgts[0], dds_config_data2(css[0]), -dds_config_len,
        data_tgts[0], dds_data2(100e6),
        -(100_000_000 - dds_headless_len + dds_data_len - 8),
        # End of step
        upd_tgts[0], 1,
        -8,
        upd_tgts[0], 0,
        ], [-200_003_000]])

    comp.runtime_finalize(8)
    test_env.check_unchanged(comp)

class DummySystem:
    pass

class DummyDaxModule(dummy_artiq.DummyDaxSystem):
    @rtval.CompositeRTProp
    def cprop(self):
        return [self.rt_dataset('ACB'), self.rt_dataset('BAC')]

def test_rt_value():
    sys = dummy_artiq.DummyDaxSystem()
    submod = DummyDaxModule()
    sys.register_child(submod)

    x = 1.2
    fx = lambda: x
    rx = sys.rt_value(fx)
    assert len(sys._bb_rt_values) == 1
    assert fx in sys._bb_rt_values
    assert str(rx) == f'({fx})()'
    rd1 = sys.rt_dataset('XXX')
    assert str(rd1) == f'<dataset XXX for {sys}>'
    assert len(sys._bb_rt_values) == 2
    assert ('XXX', False) in sys._bb_rt_values
    assert str(sys.rt_dataset('XXX')) == f'<dataset XXX for {sys}>'
    assert len(sys._bb_rt_values) == 2
    assert ('XXX', False) in sys._bb_rt_values
    sys.set_dataset('XXX', 2)
    sys.rt_dataset_sys('XXX')
    assert len(sys._bb_rt_values) == 3
    assert ('XXX', True) in sys._bb_rt_values
    sys.set_dataset_sys('XXX', 3)

    rd2 = submod.rt_dataset_sys('XXX')
    assert str(rd2) == f'<dataset_sys XXX for {submod}>'
    assert len(submod._bb_rt_values) == 1
    assert ('XXX', True) in submod._bb_rt_values
    assert str(submod.rt_dataset_sys('XXX')) == f'<dataset_sys XXX for {submod}>'
    assert len(submod._bb_rt_values) == 1
    assert ('XXX', True) in submod._bb_rt_values
    submod.set_dataset_sys('XXX', 4)
    rd3 = submod.rt_dataset_sys('YYY', 1.2)
    assert len(submod._bb_rt_values) == 2
    assert ('YYY', True) in submod._bb_rt_values
    assert str(rd3) == f'<dataset_sys YYY for {submod}>'
    cprop = submod.cprop
    assert len(submod._bb_rt_values) == 4
    assert ('ACB', False) in submod._bb_rt_values
    assert ('BAC', False) in submod._bb_rt_values
    submod.set_dataset('ACB', 6)
    submod.set_dataset('BAC', 9)

    with pytest.raises(RuntimeError, match="Value evaluated too early"):
        rtval.get_value(rx, 1)
    with pytest.raises(RuntimeError, match="Value evaluated too early"):
        rtval.get_value(rd1, 1)
    with pytest.raises(RuntimeError, match="Value evaluated too early"):
        rtval.get_value(rd2, 1)
    with pytest.raises(RuntimeError, match="Value evaluated too early"):
        rtval.get_value(rd3, 1)
    with pytest.raises(RuntimeError, match="Value evaluated too early"):
        rtval.get_value(cprop[0], 1)
    with pytest.raises(RuntimeError, match="Value evaluated too early"):
        rtval.get_value(cprop[1], 1)

    sys._eval_all_rtvals()
    sys._eval_all_rtvals()

    x = 2.3
    assert rtval.get_value(rx, 2) == 1.2
    assert sys.rt_value(fx) == 2.3
    sys.set_dataset('XXX', 3)
    assert rtval.get_value(rd1, 2) == 2
    assert sys.rt_dataset('XXX') == 3
    submod.set_dataset_sys('XXX', 2)
    assert rtval.get_value(rd2, 2) == 4
    assert submod.rt_dataset_sys('XXX') == 2
    submod.set_dataset_sys('YYY', 2.3)
    assert rtval.get_value(rd3, 2) == 1.2
    assert submod.rt_dataset_sys('YYY') == 2.3

    assert rtval.get_value(cprop[0], 1) == 6
    assert rtval.get_value(cprop[1], 1) == 9
    assert submod.cprop == [6, 9]

    sys = dummy_artiq.DummyDaxSystem()
    submod = dummy_artiq.DummyDaxSystem()
    sys.register_child(submod)

    sys._eval_all_rtvals()
    sys._eval_all_rtvals()

    sys = dummy_artiq.DummyDaxSystem()
    submod = dummy_artiq.DummyDaxSystem()
    sys.register_child(submod)

    sys._bb_rt_values = {1: 2}

    with pytest.raises(RuntimeError, match="Unknown object in runtime callbacks"):
        sys._eval_all_rtvals()

    sys = dummy_artiq.DummyDaxSystem()
    submod = dummy_artiq.DummyDaxSystem()
    sys.register_child(submod)

    with pytest.raises(TypeError):
        submod.rt_dataset_sys()
    with pytest.raises(TypeError):
        submod.rt_dataset()

    sys = DummySystem()
    sys.get_dataset = lambda *args, **kwargs: 1
    v = dummy_artiq.DummyDaxSystem.rt_dataset(sys, 'abc')
    assert str(v) == f'<dataset abc for {sys.get_dataset}>'

def test_seq_var():
    sys = dummy_artiq.DummyDaxSystem()
    v1 = sys.seq_variable()
    v2 = sys.seq_variable(2.3)
    assert v1.eval(0) == 0.0
    assert v2.eval(0) == 2.3
    assert v1.value == 0.0
    assert v2.value == 2.3
    assert str(v1) == repr(v1) == "var(0.0)"
    assert str(v2) == repr(v2) == "var(2.3)"

    v1.value = 1.1
    v2.value = -2.0
    assert str(v1) == repr(v1) == "var(1.1)"
    assert str(v2) == repr(v2) == "var(-2.0)"
    assert v1.eval(0) == 0.0
    assert v2.eval(0) == 2.3
    assert v1.value == 1.1
    assert v2.value == -2.0
    assert v1.eval(1) == 1.1
    assert v2.eval(1) == -2.0
