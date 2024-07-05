#

import dummy_artiq
dummy_artiq.inject()

from brassboard_seq.config import Config
from brassboard_seq import action, artiq_backend, backend, seq, rtval
import brassboard_seq_test_utils as test_utils
import brassboard_seq_artiq_backend_utils as artiq_utils
import pytest
import numpy as np

global_conf = Config()
global_conf.add_supported_prefix('artiq')
global_conf.add_supported_prefix('rfsoc')

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
    return int(phase * 0x10000 + 0.5) & 0xffff

def dds_freq_to_mu(freq):
    return int(freq * 4.294967296 + 0.5)

def new_seq_compiler(*args):
    s = seq.Seq(global_conf, *args)
    comp = backend.SeqCompiler(s)
    return s, comp

def add_artiq_backend(comp, sys):
    ab = artiq_backend.ArtiqBackend(sys)
    comp.add_backend('artiq', ab)
    comp.add_backend('rfsoc', backend.Backend()) # Dummy backend
    return ab

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

with_artiq_params = with_params((0,), (5,), (500,))

def test_constructors():
    with pytest.raises(TypeError):
        artiq_backend.ArtiqBackend()

def get_channel_info(ab):
    channels = artiq_utils.get_channel_info(ab)
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

def get_compiled_info(ab, s):
    compiled_info = artiq_utils.get_compiled_info(ab)
    channels = artiq_utils.get_channel_info(ab)
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
    for artiq_action in compiled_info.all_actions:
        chn, action, seen = all_actions[artiq_action.aid]
        action_info = test_utils.action_get_compile_info(action)
        assert test_utils.action_get_exact_time(action) == artiq_action.exact_time
        if artiq_action.reloc_id >= 0:
            reloc = compiled_info.relocations[artiq_action.reloc_id]
            relocations_used[artiq_action.reloc_id] = True
            assert reloc.cond_idx >= 0 or reloc.val_idx >= 0 or reloc.time_idx >= 0
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

        event_time = test_utils.seq_get_event_time(s, artiq_action.tid)
        static_time = test_utils.event_time_get_static(event_time)
        if time_idx >= 0:
            assert static_time == -1
        else:
            assert seq_time_to_mu(static_time) == artiq_action.time_mu

        if artiq_action.tid == action_info['tid']:
            assert not seen[0]
            seen[0] = True
            value = test_utils.action_get_value(action)
        else:
            assert artiq_action.tid == action_info['end_tid']
            assert test_utils.action_get_is_pulse(action)
            assert not seen[1]
            seen[1] = True
            value = action_info['end_val']
        # Check channel
        if chn in channels.ttl_chn_map:
            is_ttl = True
            ttlid = channels.ttl_chn_map[chn]
            assert ttlid == artiq_action.chn_idx
            ttlchn = channels.ttlchns[ttlid]
            if ttlchn.iscounter:
                assert artiq_action.type == 'counter'
            else:
                assert artiq_action.type == 'ttl'
        else:
            is_ttl = False
            assert chn in channels.dds_param_chn_map
            ddsid, chntype = channels.dds_param_chn_map[chn]
            assert artiq_action.type == artiq_utils.channel_type_names[chntype]
            assert artiq_action.chn_idx == ddsid
        # Check value
        if val_idx >= 0:
            assert isinstance(value, rtval.RuntimeValue)
            if is_ttl:
                bool_values_used[val_idx] = True
                assert compiled_info.bool_values[val_idx] is value
            else:
                float_values_used[val_idx] = True
                assert compiled_info.float_values[val_idx] is value
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
        if test_utils.action_get_is_pulse(action):
            assert seen[1]
        else:
            assert not seen[1]
    assert all(bool_values_used)
    assert all(float_values_used)
    assert all(relocations_used)
    return compiled_info

@with_artiq_params
def test_val_error(max_bt):
    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.DummyDaxSystem())
    # This causes a error to be thrown when converting to boolean
    def ajquoeiauhfasd():
        s.set('artiq/ttl0', np.array([1, 2]))
    ajquoeiauhfasd()
    with pytest.raises(ValueError) as exc:
        comp.finalize()
    check_bt(exc, max_bt, 'ajquoeiauhfasd')

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.DummyDaxSystem())
    # This causes a error to be thrown when converting to float
    def jaus7hfas9dafs():
        s.set('artiq/urukul3_ch0/freq', [1, 2])
    jaus7hfas9dafs()
    with pytest.raises(TypeError) as exc:
        comp.finalize()
    check_bt(exc, max_bt, 'jaus7hfas9dafs')

@with_artiq_params
def test_channels(max_bt):
    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.DummyDaxSystem())
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
    channels = get_channel_info(ab)
    assert channels.ttl_chn_map == {0: 0, 1: 1, 2: 2, 7: 3, 8: 4}
    assert not channels.ttlchns[0].iscounter
    assert not channels.ttlchns[1].iscounter
    assert channels.ttlchns[2].iscounter
    assert channels.dds_param_chn_map == {4: (0, 0), 5: (0, 1), 6: (1, 2)}
    compiled_info = get_compiled_info(ab, s)
    assert not compiled_info.all_actions
    assert not compiled_info.bool_values
    assert not compiled_info.float_values
    assert not compiled_info.relocations

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.HasEnvironment())
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
    channels = get_channel_info(ab)
    assert channels.ttl_chn_map == {0: 0, 2: 1, 4: 2, 7: 3, 8: 4}
    assert not channels.ttlchns[0].iscounter
    assert not channels.ttlchns[1].iscounter
    assert channels.ttlchns[2].iscounter
    assert channels.dds_param_chn_map == {1: (0, 0), 3: (0, 1), 5: (1, 2)}
    compiled_info = get_compiled_info(ab, s)
    assert not compiled_info.all_actions
    assert not compiled_info.bool_values
    assert not compiled_info.float_values
    assert not compiled_info.relocations

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.HasEnvironment())
    s.get_channel_id('artiq/urukul0_ch0/a/b')
    with pytest.raises(ValueError, match="Invalid channel name artiq/urukul0_ch0/a/b"):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.HasEnvironment())
    s.get_channel_id('artiq/urukul0_ch0/aaa')
    with pytest.raises(ValueError, match="Invalid channel name artiq/urukul0_ch0/aaa"):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.HasEnvironment())
    s.get_channel_id('artiq/ttl0/a')
    with pytest.raises(ValueError, match="Invalid channel name artiq/ttl0/a"):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.HasEnvironment())
    s.get_channel_id('artiq/ttl0_counter/a')
    with pytest.raises(ValueError, match="Invalid channel name artiq/ttl0_counter/a"):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.HasEnvironment())
    s.get_channel_id('artiq/something')
    with pytest.raises(Exception):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.HasEnvironment())
    s.get_channel_id('artiq')
    with pytest.raises(ValueError, match="Invalid channel name artiq"):
        comp.finalize()

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.HasEnvironment())
    s.get_channel_id('artiq/dummy_dev')
    with pytest.raises(ValueError, match="Unsupported device: "):
        comp.finalize()

@with_artiq_params
def test_action_kws(max_bt):
    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.DummyDaxSystem())
    def ashdfjasldfas2a7sdfds(step):
        step.set('artiq/ttl0', True, arg=123)
    ashdfjasldfas2a7sdfds(s.add_step(0.01))
    with pytest.raises(ValueError, match="Invalid output keyword argument") as exc:
        comp.finalize()
    check_bt(exc, max_bt, 'ashdfjasldfas2a7sdfds')

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.DummyDaxSystem())
    def asdf89hsdf():
        s.add_step(0.01) \
         .set('artiq/ttl0_counter', True, b=True)
    asdf89hsdf()
    with pytest.raises(ValueError, match="Invalid output keyword argument") as exc:
        comp.finalize()
    check_bt(exc, max_bt, 'asdf89hsdf')

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.DummyDaxSystem())
    def ahsd87hfasdf():
        s.add_step(0.01) \
         .set('artiq/urukul0_ch1/freq', 1.2e7, b=True)
    ahsd87hfasdf()
    with pytest.raises(ValueError, match="Invalid output keyword argument") as exc:
        comp.finalize()
    check_bt(exc, max_bt, 'ahsd87hfasdf')

@with_artiq_params
def test_ttl(max_bt):
    v = rtval.new_extern(lambda: True)
    v2 = rtval.new_extern(lambda: 0.2)
    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.DummyDaxSystem())
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
    compiled_info = get_compiled_info(ab, s)
    assert len(compiled_info.all_actions) == 8

@with_artiq_params
def test_counter(max_bt):
    v = rtval.new_extern(lambda: True)
    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.DummyDaxSystem())
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
    compiled_info = get_compiled_info(ab, s)
    assert len(compiled_info.all_actions) == 9

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.DummyDaxSystem())
    def ahsd87fasdfasf():
        s.add_step(0.01) \
         .set('artiq/ttl0_counter', v)
    ahsd87fasdfasf()
    with pytest.raises(ValueError, match="Counter value must be static") as exc:
        comp.finalize()
    check_bt(exc, max_bt, 'ahsd87fasdfasf')

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.DummyDaxSystem())
    s.conditional(v).add_step(0.01) \
     .set('artiq/ttl0_counter', True)
    s.wait(1e-6)
    def as78j3df89asdf():
        s.add_step(0.01) \
         .pulse('artiq/ttl0_counter', True)
    as78j3df89asdf()
    with pytest.raises(ValueError, match="Counter value must be static") as exc:
        comp.finalize()
    check_bt(exc, max_bt, 'as78j3df89asdf')

@with_artiq_params
def test_dds(max_bt):
    v = rtval.new_extern(lambda: 123e6)
    v2 = rtval.new_extern(lambda: 0.1)
    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.DummyDaxSystem())
    @s.add_step
    def step(s):
        s.add_step(0.01) \
         .set('artiq/ttl0', True) \
         .pulse('artiq/urukul0_ch0/amp', v2) \
         .set('artiq/urukul0_ch0/phase', v2) \
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
    compiled_info = get_compiled_info(ab, s)
    assert len(compiled_info.all_actions) == 14

class StaticFunction(action.RampFunction):
    def __init__(self):
        action.RampFunction.__init__(self)

    def eval(self, t, length, oldval):
        return t / 2 + oldval - length

@with_artiq_params
def test_ramp(max_bt):
    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.DummyDaxSystem())
    def sdf67asdf7f8sdf():
        s.add_step(0.01) \
         .set('artiq/ttl0', StaticFunction())
    sdf67asdf7f8sdf()
    with pytest.raises(ValueError, match="TTL Channel cannot be ramped") as exc:
        comp.finalize()
    check_bt(exc, max_bt, 'sdf67asdf7f8sdf')

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.DummyDaxSystem())
    def asdf787asd78hte78():
        s.add_step(0.01) \
         .set('artiq/ttl0_counter', StaticFunction())
    asdf787asd78hte78()
    with pytest.raises(ValueError, match="TTL Channel cannot be ramped") as exc:
        comp.finalize()
    check_bt(exc, max_bt, 'asdf787asd78hte78')

    s, comp = new_seq_compiler(max_bt)
    ab = add_artiq_backend(comp, dummy_artiq.DummyDaxSystem())
    def as8df9sdf8():
        s.add_step(0.01) \
         .set('artiq/urukul0_ch2/amp', StaticFunction())
    as8df9sdf8()
    with pytest.raises(ValueError, match="DDS Channel cannot be ramped") as exc:
        comp.finalize()
    check_bt(exc, max_bt, 'as8df9sdf8')
