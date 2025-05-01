#

from brassboard_seq import rfsoc_backend
from brassboard_seq.rfsoc_backend import Jaqal_v1_3, JaqalInst_v1_3

import pytest
import re
import random
import itertools

def channels_str(d):
    chns = d['channels']
    assert all(chn < 8 and chn >= 0 for chn in chns)
    assert len(set(chns)) == len(chns)
    if len(chns) == 1:
        assert d['channel'] == chns[0]
        return f'.{chns[0]}'
    else:
        assert 'channel' not in d
    if len(chns) == 0:
        return '{}'
    if len(chns) == 8:
        return '.all'
    return '{' + ','.join(str(chn) for chn in chns) + '}'

_modtype_names = ['freq0', 'amp0', 'phase0', 'freq1',
                  'amp1', 'phase1', 'frame_rot0', 'frame_rot1']
def modtype_name_list(mask):
    return [_modtype_names[i] for i in range(8) if (mask >> i) & 1]

class MatchGLUT:
    def __init__(self, channels=None, gaddrs=None, starts=None, ends=None, cnt=None):
        self.channels = channels
        self.gaddrs = list(gaddrs) if gaddrs is not None else None
        self.starts = list(starts) if starts is not None else None
        self.ends = list(ends) if ends is not None else None
        if self.gaddrs is not None:
            assert cnt is None or cnt == len(self.gaddrs)
            cnt = len(self.gaddrs)
        if self.starts is not None:
            assert cnt is None or cnt == len(self.starts)
            cnt = len(self.starts)
        if self.ends is not None:
            assert cnt is None or cnt == len(self.ends)
            cnt = len(self.ends)
        self.cnt = cnt

    def __eq__(self, other):
        if not isinstance(other, JaqalInst_v1_3):
            return NotImplemented
        d = other.to_dict()
        assert d['type'] == 'glut'
        lut_str = ''.join(f' [{gaddr}]=[{start},{end}]' for gaddr, start, end in
                            zip(d['gaddrs'], d['starts'], d['ends']))
        assert str(other) == f"glut{channels_str(d)}[{d['count']}]{lut_str}"
        if self.channels is not None:
            assert d['channels'] == self.channels
        if self.cnt is not None:
            assert d['count'] == self.cnt
        if self.gaddrs is not None:
            assert d['gaddrs'] == self.gaddrs
        if self.starts is not None:
            assert d['starts'] == self.starts
        if self.ends is not None:
            assert d['ends'] == self.ends
        return True

class MatchSLUT:
    def __init__(self, channels=None, saddrs=None, paddrs=None, modtypes=None, cnt=None):
        self.channels = channels
        self.saddrs = list(saddrs) if saddrs is not None else None
        self.paddrs = list(paddrs) if paddrs is not None else None
        self.modtypes = list(modtypes) if modtypes is not None else None
        if self.saddrs is not None:
            assert cnt is None or cnt == len(self.saddrs)
            cnt = len(self.saddrs)
        if self.paddrs is not None:
            assert cnt is None or cnt == len(self.paddrs)
            cnt = len(self.paddrs)
        if self.modtypes is not None:
            assert cnt is None or cnt == len(self.modtypes)
            cnt = len(self.modtypes)
        self.cnt = cnt

    def __eq__(self, other):
        if not isinstance(other, JaqalInst_v1_3):
            return NotImplemented
        d = other.to_dict()
        assert d['type'] == 'slut'
        lut_str = ''.join(f" [{saddr}]={paddr}{{{','.join(modtype)}}}"
                            for saddr, paddr, modtype in
                            zip(d['saddrs'], d['paddrs'], d['modtypes']))
        assert str(other) == f"slut{channels_str(d)}[{d['count']}]{lut_str}"
        if self.channels is not None:
            assert d['channels'] == self.channels
        if self.cnt is not None:
            assert d['count'] == self.cnt
        if self.saddrs is not None:
            assert d['saddrs'] == self.saddrs
        if self.paddrs is not None:
            assert d['paddrs'] == self.paddrs
        if self.modtypes is not None:
            assert d['modtypes'] == self.modtypes
        return True

class MatchGSEQ:
    def __init__(self, channels=None, mode=None, gaddrs=None, cnt=None):
        self.channels = channels
        self.gaddrs = list(gaddrs) if gaddrs is not None else None
        self.mode = mode
        if self.gaddrs is not None:
            assert cnt is None or cnt == len(self.gaddrs)
            cnt = len(self.gaddrs)
        self.cnt = cnt

    def __eq__(self, other):
        if not isinstance(other, JaqalInst_v1_3):
            return NotImplemented
        d = other.to_dict()
        seq_str = ''.join(f' {gaddr}' for gaddr in d['gaddrs'])
        assert str(other) == f"{d['type']}{channels_str(d)}[{d['count']}]{seq_str}"
        if self.mode is not None:
            assert d['type'] == self.mode
        if self.channels is not None:
            assert d['channels'] == self.channels
        if self.cnt is not None:
            assert d['count'] == self.cnt
        if self.gaddrs is not None:
            assert d['gaddrs'] == self.gaddrs
        return True

def pad_list(lst, n, v):
    return lst + [v] * (n - len(lst))

class MatchPulse:
    def __init__(self, mode, params, channels=None, addr=None, cycles=None,
                 ispl=None, shift=None,
                 spl_freq=None, spl_amp=None, spl_phase=None, rel=None, abs=None,
                 trig=None, sync=None, enable=None, ff=None, eof=None, clr=None,
                 fwd=None, inv=None):
        assert mode is not None
        assert params is not None
        self.mode = mode
        self.params = params

        self.channels = channels
        self.addr = addr
        if addr is not None:
            assert mode == 'plut'
        self.cycles = cycles
        self.ispl = ispl
        self.shift = shift
        self.spl_freq = spl_freq
        self.spl_amp = spl_amp
        self.spl_phase = spl_phase
        self.rel = rel
        self.abs = abs
        self.trig = trig
        self.sync = sync
        self.enable = enable
        self.ff = ff
        self.eof = eof
        self.clr = clr
        self.fwd = fwd
        self.inv = inv

    def test_str(self, d, inst):
        str_inst = str(inst)
        repr_inst = repr(inst)
        m_str = re.match('^(.*<[0-9]*>) (?:{([^{}]+)}|(?:(.*{[^{}]+}){1,3}))([^{}]*)$', str_inst)
        assert m_str is not None
        m_repr = re.match('^(.*<[0-9]*>) {([^{}]+)}([^{}]*)$', repr_inst)
        assert m_repr is not None
        assert m_str[1] == m_repr[1]
        assert m_str[4] == m_repr[3]

        prefix = f"{d['type']}{channels_str(d)} "
        is_plut = d['type'] == 'plut'
        if is_plut:
            prefix += f"[{d['paddr']}]={{}}"
        else:
            modtype = d['modtype']
            if len(modtype) == 1:
                prefix += modtype[0]
            else:
                prefix += '{' + ','.join(modtype) + '}'
        prefix += f" <{d['cycles']}>"
        assert m_str[1] == prefix

        spline_mu = d['spline_mu']
        spline_shift = d['spline_shift']
        ispl_str = ''
        found_order = False
        for order in range(3, -1, -1):
            spl_order = spline_mu[order]
            if not (found_order or spl_order or order == 0):
                continue
            found_order = True
            if spl_order == 0:
                order_str = '0'
            else:
                if spl_order < 0:
                    spl_order += 2**64
                    assert spl_order > 0
                order_str = hex(spl_order)
            if spline_shift * order:
                ispl_str = f">>{spline_shift * order}" + ispl_str
            ispl_str = f"{order_str}" + ispl_str
            if order != 0:
                ispl_str = ", " + ispl_str
        assert m_repr[2] == ispl_str

        if is_plut or not d['modtype']:
            assert str_inst == repr_inst
        else:
            has_freq_spl = bool({'freq0', 'freq1'}.intersection(d['modtype']))
            has_amp_spl = bool({'amp0', 'amp1'}.intersection(d['modtype']))
            has_phase_spl = bool({'phase0', 'frame_rot0',
                                  'phase1', 'frame_rot1'}.intersection(d['modtype']))
            num_spls = has_freq_spl + has_amp_spl + has_phase_spl
            assert num_spls
            splines = {}
            if num_spls == 1:
                assert m_str[2]
                if has_freq_spl:
                    splines['freq'] = m_str[2]
                elif has_amp_spl:
                    splines['amp'] = m_str[2]
                else:
                    assert has_phase_spl
                    splines['phase'] = m_str[2]
            else:
                assert m_str[3]
                for spl_str in m_str[3].split('} '):
                    param, values = spl_str.split('{')
                    splines[param.strip()] = values.strip('}')
            splines = {param: pad_list([float(o) for o in s.split(', ')], 4, 0)
                       for param, s in splines.items()}
            if has_freq_spl:
                assert splines['freq'] == d['spline_freq']
            if has_amp_spl:
                assert splines['amp'] == d['spline_amp']
            if has_phase_spl:
                assert splines['phase'] == d['spline_phase']

        flags = ' trig' if d['trig'] else ''
        if is_plut or {'freq0', 'amp0', 'phase0',
                       'freq1', 'amp1', 'phase1'}.intersection(d['modtype']):
            if d['sync']:
                flags += ' sync'
            if d['enable']:
                flags += ' enable'
            if d['ff']:
                flags += ' ff'
        if is_plut or {'frame_rot0', 'frame_rot1'}.intersection(d['modtype']):
            if d['eof']:
                flags += ' eof'
            if d['clr']:
                flags += ' clr'
            flags += f" fwd:{d['fwd']} inv:{d['inv']}"
        assert m_str[4] == flags

    def __eq__(self, other):
        if not isinstance(other, JaqalInst_v1_3):
            return NotImplemented
        d = other.to_dict()
        self.test_str(d, other)

        assert d['type'] == self.mode
        is_plut = d['type'] == 'plut'
        if is_plut:
            assert 'modtype' not in d
        else:
            assert d['modtype'] == self.params
        if self.channels is not None:
            assert d['channels'] == self.channels
        if self.addr is not None:
            assert d['paddr'] == self.addr
        if self.cycles is not None:
            assert d['cycles'] == self.cycles

        if self.ispl is not None:
            assert d['spline_mu'] == pad_list(list(self.ispl), 4, 0)
        if self.shift is not None:
            assert d['spline_shift'] == self.shift

        def test_fspl(name):
            fld_name = 'spline_' + name
            expected = getattr(self, fld_name)
            if expected is None:
                return
            expected = pad_list(list(expected), 4, 0)
            if self.abs is not None or self.rel is not None:
                expected = pytest.approx(expected, abs=self.abs, rel=self.rel)
            assert d[fld_name] == expected

        if self.trig is not None:
            assert d['trig'] == self.trig
        if self.sync is not None:
            assert d['sync'] == self.sync
        if self.enable is not None:
            assert d['enable'] == self.enable
        if self.ff is not None:
            assert d['ff'] == self.ff
        if self.trig is not None:
            assert d['trig'] == self.trig
        if self.eof is not None:
            assert d['eof'] == self.eof
        if self.clr is not None:
            assert d['clr'] == self.clr
        if self.fwd is not None:
            assert d['fwd'] == self.fwd
        if self.inv is not None:
            assert d['inv'] == self.inv

        return True

invalid_pulse = JaqalInst_v1_3(b'\xff' * 32)

def check_invalid(v, msg):
    inst_bytes = v.to_bytes(32, 'little')
    assert Jaqal_v1_3.dump_insts(inst_bytes) == f'invalid({msg}): {v:0>64x}'
    d = JaqalInst_v1_3(inst_bytes).to_dict()
    assert d['type'] == 'invalid'
    assert d['error'] == msg
    assert d['inst'] == f'{v:0>64x}'
    pulses, = Jaqal_v1_3.extract_pulses(inst_bytes)
    assert pulses == invalid_pulse


def test_insts():
    assert Jaqal_v1_3.dump_insts(b'') == ''
    with pytest.raises(ValueError, match="Instruction stream length not a multiple of instruction size"):
        Jaqal_v1_3.dump_insts(b'x')

    assert rfsoc_backend.JaqalInst_v1() != JaqalInst_v1_3()

    assert str(JaqalInst_v1_3()) == 'pulse_data{} {} <0> {0}'
    assert JaqalInst_v1_3().to_dict() == {'type': 'pulse_data', 'modtype': [],
                                          'channels': [], 'cycles': 0,
                                          'spline_mu': [0, 0, 0, 0], 'spline_shift': 0,
                                          'trig': False, 'sync': False, 'enable': False,
                                          'ff': False, 'eof': False, 'clr': False,
                                          'fwd': 0, 'inv': 0}
    with pytest.raises(TypeError, match=f"Unexpected type '{int}' for data"):
        JaqalInst_v1_3(2)
    with pytest.raises(ValueError, match=f"Invalid address '-1'"):
        Jaqal_v1_3.program_PLUT(JaqalInst_v1_3(), -1)
    with pytest.raises(ValueError, match=f"Invalid address '4096'"):
        Jaqal_v1_3.program_PLUT(JaqalInst_v1_3(), 4096)

def test_glut():
    with pytest.raises(ValueError, match="Invalid channel number '-1'"):
        Jaqal_v1_3.program_GLUT(-1, [], [], [])
    with pytest.raises(ValueError, match="Invalid channel number '8'"):
        Jaqal_v1_3.program_GLUT(8, [], [], [])
    with pytest.raises(ValueError, match="Invalid channel number '-1'"):
        Jaqal_v1_3.program_GLUT((-1, 8), [], [], [])
    with pytest.raises(ValueError, match="Invalid channel number '8'"):
        Jaqal_v1_3.program_GLUT((5, 8), [], [], [])
    with pytest.raises(ValueError, match="Mismatch address length"):
        Jaqal_v1_3.program_GLUT(3, [1], [2, 3], [5])
    with pytest.raises(ValueError, match="Mismatch address length"):
        Jaqal_v1_3.program_GLUT(2, [1], [3], [5, 6])
    with pytest.raises(ValueError, match="Mismatch address length"):
        Jaqal_v1_3.program_GLUT(1, [1, 2], [3], [5])
    with pytest.raises(ValueError, match="Too many GLUT addresses to program"):
        Jaqal_v1_3.program_GLUT(0, list(range(7)), list(range(7)), list(range(7)))
    for n in range(7):
        for i in range(2000):
            chn_mask = random.randint(0, 255)
            chns = [chn for chn in range(8) if (chn_mask >> chn) & 1]
            gaddrs = [random.randint(0, 4095) for _ in range(n)]
            starts = [random.randint(0, 8191) for _ in range(n)]
            ends = [random.randint(0, 8191) for _ in range(n)]
            inst = Jaqal_v1_3.program_GLUT(chns, gaddrs, starts, ends)
            if len(chns) == 1:
                assert Jaqal_v1_3.program_GLUT(chns[0], gaddrs, starts, ends) == inst
            assert inst.channel_mask == chn_mask
            assert inst.channels == chns
            assert inst == MatchGLUT(chns, gaddrs, starts, ends)
            assert str(inst) == repr(inst)
            assert Jaqal_v1_3.dump_insts(inst.to_bytes()) == str(inst)
            assert Jaqal_v1_3.dump_insts(inst.to_bytes() + inst.to_bytes()) == str(inst) + '\n' + str(inst)
            assert inst == JaqalInst_v1_3(inst.to_bytes())

    glut_base = 0x20000000000000000000000000000000000000000000000000000000000000
    for bit in itertools.chain(range(239, 245), range(228, 236)):
        check_invalid(glut_base | (1 << bit), 'reserved')
    for n in range(7):
        glut_base_n = glut_base | (n << 236)
        for bit in range(n * 38, 228):
            check_invalid(glut_base_n | (1 << bit), 'reserved')
    for n in range(7, 8):
        check_invalid(glut_base | (n << 236), 'glut_oob')

def test_slut():
    with pytest.raises(ValueError, match="Invalid channel number '-1'"):
        Jaqal_v1_3.program_SLUT(-1, [], [], [])
    with pytest.raises(ValueError, match="Invalid channel number '8'"):
        Jaqal_v1_3.program_SLUT(8, [], [], [])
    with pytest.raises(ValueError, match="Invalid channel number '-1'"):
        Jaqal_v1_3.program_SLUT((-1, 8), [], [], [])
    with pytest.raises(ValueError, match="Invalid channel number '8'"):
        Jaqal_v1_3.program_SLUT((5, 8), [], [], [])
    with pytest.raises(ValueError, match="Mismatch address length"):
        Jaqal_v1_3.program_SLUT(3, [1], [2, 3], [2])
    with pytest.raises(ValueError, match="Invalid mod type '277'"):
        Jaqal_v1_3.program_SLUT(3, [1], [2], [277])
    with pytest.raises(ValueError, match="Invalid mod type '-3'"):
        Jaqal_v1_3.program_SLUT(2, [1, 2], [2, 3], [2, -3])
    with pytest.raises(ValueError, match="Invalid mod type 'f2'"):
        Jaqal_v1_3.program_SLUT(3, [1], [2], ['f2'])
    with pytest.raises(ValueError, match="Invalid mod type 'xyz'"):
        Jaqal_v1_3.program_SLUT(2, [1, 2], [2, 3], [2, ('freq0', 'xyz')])
    with pytest.raises(ValueError, match="Too many SLUT addresses to program"):
        Jaqal_v1_3.program_SLUT(0, list(range(7)), list(range(7)), list(range(7)))
    for n in range(7):
        for i in range(3000):
            chn_mask = random.randint(0, 255)
            chns = [chn for chn in range(8) if (chn_mask >> chn) & 1]
            saddrs = [random.randint(0, 8191) for _ in range(n)]
            paddrs = [random.randint(0, 4095) for _ in range(n)]
            modtypes = [random.randint(0, 255) for _ in range(n)]
            modtype_names = [modtype_name_list(modtype) for modtype in modtypes]
            inst = Jaqal_v1_3.program_SLUT(chns, saddrs, paddrs, modtypes)
            if len(chns) == 1:
                assert Jaqal_v1_3.program_SLUT(chns[0], saddrs, paddrs, modtypes) == inst
            assert Jaqal_v1_3.program_SLUT(chns, saddrs, paddrs, modtype_names) == inst
            assert inst.channel_mask == chn_mask
            assert inst.channels == chns
            assert inst == MatchSLUT(chns, saddrs, paddrs, modtype_names)
            assert str(inst) == repr(inst)
            assert Jaqal_v1_3.dump_insts(inst.to_bytes()) == str(inst)
            assert Jaqal_v1_3.dump_insts(inst.to_bytes() + inst.to_bytes()) == str(inst) + '\n' + str(inst)
            assert inst == JaqalInst_v1_3(inst.to_bytes())

    slut_base = 0x40000000000000000000000000000000000000000000000000000000000000
    for bit in itertools.chain(range(239, 245), range(198, 236)):
        check_invalid(slut_base | (1 << bit), 'reserved')
    for n in range(7):
        slut_base_n = slut_base | (n << 236)
        for bit in range(n * 33, 198):
            check_invalid(slut_base_n | (1 << bit), 'reserved')
    for n in range(7, 8):
        check_invalid(slut_base | (n << 236), 'slut_oob')

def test_gseq():
    with pytest.raises(ValueError, match="Invalid channel number '-1'"):
        Jaqal_v1_3.sequence(-1, 0, [])
    with pytest.raises(ValueError, match="Invalid channel number '8'"):
        Jaqal_v1_3.sequence(8, 1, [])
    with pytest.raises(ValueError, match="Invalid channel number '-1'"):
        Jaqal_v1_3.sequence((-1, 8), 0, [])
    with pytest.raises(ValueError, match="Invalid channel number '8'"):
        Jaqal_v1_3.sequence((3, 8), 1, [])
    with pytest.raises(ValueError, match="Invalid sequencing mode 3"):
        Jaqal_v1_3.sequence(2, 3, [])
    with pytest.raises(ValueError, match="Too many GLUT addresses to sequence"):
        Jaqal_v1_3.sequence(3, 2, list(range(21)))
    for n in range(21):
        for i in range(2000):
            chn_mask = random.randint(0, 255)
            chns = [chn for chn in range(8) if (chn_mask >> chn) & 1]
            mode = random.randint(0, 2)
            mode_str = ('gseq', 'wait_anc', 'cont_anc')[mode]
            gaddrs = [random.randint(0, 2047) for _ in range(n)]
            inst = Jaqal_v1_3.sequence(chns, mode, gaddrs)
            if len(chns) == 1:
                assert Jaqal_v1_3.sequence(chns[0], mode, gaddrs) == inst
            assert inst.channel_mask == chn_mask
            assert inst.channels == chns
            assert inst == MatchGSEQ(chns, mode_str, gaddrs)
            assert str(inst) == repr(inst)
            assert Jaqal_v1_3.dump_insts(inst.to_bytes()) == str(inst)
            assert Jaqal_v1_3.dump_insts(inst.to_bytes() + inst.to_bytes()) == str(inst) + '\n' + str(inst)
            assert inst == JaqalInst_v1_3(inst.to_bytes())

    gseq_base = 0x80000000000000000000000000000000000000000000000000000000000000
    for bit in itertools.chain(range(244, 245), range(220, 239)):
        check_invalid(gseq_base | (1 << bit), 'reserved')
    for n in range(21):
        gseq_base_n = gseq_base | (n << 239)
        for bit in range(n * 11, 220):
            check_invalid(gseq_base_n | (1 << bit), 'reserved')
    for n in range(21, 32):
        check_invalid(gseq_base | (n << 239), 'gseq_oob')

def test_pulse_inst():
    with pytest.raises(ValueError, match="Invalid channel number '-1'"):
        Jaqal_v1_3.apply_channel_mask(JaqalInst_v1_3(), -1)
    with pytest.raises(ValueError, match="Invalid channel number '8'"):
        Jaqal_v1_3.apply_channel_mask(JaqalInst_v1_3(), 8)
    with pytest.raises(ValueError, match="Invalid channel number '-1'"):
        Jaqal_v1_3.apply_channel_mask(JaqalInst_v1_3(), (-1, 3))
    with pytest.raises(ValueError, match="Invalid channel number '8'"):
        Jaqal_v1_3.apply_channel_mask(JaqalInst_v1_3(), (2, 8))
    with pytest.raises(ValueError, match="Invalid mod type '277'"):
        Jaqal_v1_3.apply_modtype_mask(JaqalInst_v1_3(), 277)
    with pytest.raises(ValueError, match="Invalid mod type '-3'"):
        Jaqal_v1_3.apply_modtype_mask(JaqalInst_v1_3(), -3)
    with pytest.raises(ValueError, match="Invalid mod type 'f2'"):
        Jaqal_v1_3.apply_modtype_mask(JaqalInst_v1_3(), 'f2')
    with pytest.raises(ValueError, match="Invalid mod type 'xyz'"):
        Jaqal_v1_3.apply_modtype_mask(JaqalInst_v1_3(), ('freq0', 'xyz'))

    with pytest.raises(TypeError, match=f"Unexpected type '{int}' for spline"):
        Jaqal_v1_3.freq_pulse(0, 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid spline \\(0, 0, 0, 0, 0\\)"):
        Jaqal_v1_3.freq_pulse((0, 0, 0, 0, 0), 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid cycle count '1099511627776'"):
        Jaqal_v1_3.freq_pulse((0, 0, 0, 0), 2**40, False, False, False)
    with pytest.raises(TypeError, match=f"Unexpected type '{int}' for spline"):
        Jaqal_v1_3.amp_pulse(0, 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid spline \\(0, 0, 0, 0, 0\\)"):
        Jaqal_v1_3.amp_pulse((0, 0, 0, 0, 0), 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid cycle count '1099511627776'"):
        Jaqal_v1_3.amp_pulse((0, 0, 0, 0), 2**40, False, False, False)
    with pytest.raises(TypeError, match=f"Unexpected type '{int}' for spline"):
        Jaqal_v1_3.phase_pulse(0, 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid spline \\(0, 0, 0, 0, 0\\)"):
        Jaqal_v1_3.phase_pulse((0, 0, 0, 0, 0), 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid cycle count '1099511627776'"):
        Jaqal_v1_3.phase_pulse((0, 0, 0, 0), 2**40, False, False, False)
    with pytest.raises(TypeError, match=f"Unexpected type '{int}' for spline"):
        Jaqal_v1_3.frame_pulse(0, 100, False, False, False, 0, 0)
    with pytest.raises(ValueError, match="Invalid spline \\(0, 0, 0, 0, 0\\)"):
        Jaqal_v1_3.frame_pulse((0, 0, 0, 0, 0), 100, False, False, False, 0, 0)
    with pytest.raises(ValueError, match="Invalid cycle count '1099511627776'"):
        Jaqal_v1_3.frame_pulse((0, 0, 0, 0), 2**40, False, False, False, 0, 0)

    assert (Jaqal_v1_3.freq_pulse((), 100, False, False, False) ==
            Jaqal_v1_3.freq_pulse((0, 0, 0, 0), 100, False, False, False))
    assert (Jaqal_v1_3.freq_pulse((20,), 100, False, False, False) ==
            Jaqal_v1_3.freq_pulse((20, 0, 0, 0), 100, False, False, False))
    assert (Jaqal_v1_3.freq_pulse((20, 30), 100, False, False, False) ==
            Jaqal_v1_3.freq_pulse((20, 30, 0, 0), 100, False, False, False))
    assert (Jaqal_v1_3.freq_pulse((20, 30, 40), 100, False, False, False) ==
            Jaqal_v1_3.freq_pulse((20, 30, 40, 0), 100, False, False, False))

    assert (Jaqal_v1_3.amp_pulse((), 100, False, False, False) ==
            Jaqal_v1_3.amp_pulse((0, 0, 0, 0), 100, False, False, False))
    assert (Jaqal_v1_3.amp_pulse((0.2,), 100, False, False, False) ==
            Jaqal_v1_3.amp_pulse((0.2, 0, 0, 0), 100, False, False, False))
    assert (Jaqal_v1_3.amp_pulse((0.2, 0.3), 100, False, False, False) ==
            Jaqal_v1_3.amp_pulse((0.2, 0.3, 0, 0), 100, False, False, False))
    assert (Jaqal_v1_3.amp_pulse((0.2, 0.3, 0.4), 100, False, False, False) ==
            Jaqal_v1_3.amp_pulse((0.2, 0.3, 0.4, 0), 100, False, False, False))

    assert (Jaqal_v1_3.phase_pulse((), 100, False, False, False) ==
            Jaqal_v1_3.phase_pulse((0, 0, 0, 0), 100, False, False, False))
    assert (Jaqal_v1_3.phase_pulse((0.2,), 100, False, False, False) ==
            Jaqal_v1_3.phase_pulse((0.2, 0, 0, 0), 100, False, False, False))
    assert (Jaqal_v1_3.phase_pulse((0.2, 0.3), 100, False, False, False) ==
            Jaqal_v1_3.phase_pulse((0.2, 0.3, 0, 0), 100, False, False, False))
    assert (Jaqal_v1_3.phase_pulse((0.2, 0.3, 0.4), 100, False, False, False) ==
            Jaqal_v1_3.phase_pulse((0.2, 0.3, 0.4, 0), 100, False, False, False))

    assert (Jaqal_v1_3.frame_pulse((), 100, False, False, False, 0, 0) ==
            Jaqal_v1_3.frame_pulse((0, 0, 0, 0), 100, False, False, False, 0, 0))
    assert (Jaqal_v1_3.frame_pulse((0.2,), 100, False, False, False, 0, 0) ==
            Jaqal_v1_3.frame_pulse((0.2, 0, 0, 0), 100, False, False, False, 0, 0))
    assert (Jaqal_v1_3.frame_pulse((0.2, 0.3), 100, False, False, False, 0, 0) ==
            Jaqal_v1_3.frame_pulse((0.2, 0.3, 0, 0), 100, False, False, False, 0, 0))
    assert (Jaqal_v1_3.frame_pulse((0.2, 0.3, 0.4), 100, False, False, False, 0, 0) ==
            Jaqal_v1_3.frame_pulse((0.2, 0.3, 0.4, 0), 100, False, False, False, 0, 0))

    metadatas = itertools.product(('freq', 'amp', 'phase', ('freq', 'amp'),
                                  ('freq', 'phase'), ('amp', 'phase')),
                                  range(1, 8, 2), (0, 1),
                                  (False, True), (False, True), (False, True),
                                  (False, True), (4, 1000000, 1099511627775))

    for params, chn, tone, trig, sync, enable, ff, cycles in metadatas:
        if isinstance(params, tuple):
            fspls = {'spl_' + p: [0, 0, 0, 0] for p in params}
            param = params[0]
            params = tuple(f'{p}{tone}' for p in params)
            param_list = list(params)
        else:
            fspls = {'spl_' + params: [0, 0, 0, 0]}
            param = params
            params = f'{params}{tone}'
            param_list = [params]
        inst = getattr(Jaqal_v1_3, param + '_pulse')((0, 0, 0, 0), cycles, trig, sync, ff)
        inst = Jaqal_v1_3.apply_channel_mask(inst, chn)
        inst = Jaqal_v1_3.apply_modtype_mask(inst, params)
        vi = int(inst)
        assert vi & (1 << 205) == 0
        if enable:
            vi |= 1 << 205
            inst = JaqalInst_v1_3(vi.to_bytes(32, 'little'))

        addr = random.randint(0, 4095)
        plut_inst = Jaqal_v1_3.program_PLUT(inst, addr)
        stm_inst = Jaqal_v1_3.stream(inst)
        stm_frm_inst = Jaqal_v1_3.apply_modtype_mask(stm_inst, param_list + ['frame_rot0'])

        assert inst.channels == [chn]
        assert plut_inst.channels == [chn]
        assert stm_inst.channels == [chn]
        assert stm_frm_inst.channels == [chn]
        frm_fspls = fspls.copy()
        frm_fspls['spl_phase'] = [0, 0, 0, 0]

        assert inst == MatchPulse('pulse_data', param_list, [chn], cycles=cycles,
                                  trig=trig, sync=sync, enable=enable, ff=ff,
                                  ispl=(0, 0, 0, 0), shift=0, **fspls)
        assert plut_inst == MatchPulse('plut', [], [chn], cycles=cycles,
                                       trig=trig, sync=sync, enable=enable, ff=ff,
                                       ispl=(0, 0, 0, 0), shift=0)
        assert stm_inst == MatchPulse('stream', param_list, [chn], cycles=cycles,
                                       trig=trig, sync=sync, enable=enable, ff=ff,
                                       ispl=(0, 0, 0, 0), shift=0, **fspls)
        assert stm_frm_inst == MatchPulse('stream', param_list + ['frame_rot0'], [chn],
                                          cycles=cycles, trig=trig, sync=sync,
                                          enable=enable, ff=ff, eof=False,
                                          clr=ff, fwd=enable * 2 + sync, inv=0,
                                          ispl=(0, 0, 0, 0), shift=0, **frm_fspls)
        assert Jaqal_v1_3.dump_insts(inst.to_bytes()) == f'invalid(reserved): {vi:0>64x}'
        assert Jaqal_v1_3.dump_insts(plut_inst.to_bytes()) == str(plut_inst)
        assert Jaqal_v1_3.dump_insts(stm_inst.to_bytes()) == str(stm_inst)
        assert Jaqal_v1_3.dump_insts(stm_frm_inst.to_bytes()) == str(stm_frm_inst)
        assert Jaqal_v1_3.dump_insts(inst.to_bytes() + stm_inst.to_bytes() + plut_inst.to_bytes()) == f'invalid(reserved): {vi:0>64x}' + '\n' + str(stm_inst) + '\n' + str(plut_inst)

        plut_vi = int(plut_inst)
        stm_vi = int(stm_inst)
        stm_frm_vi = int(stm_frm_inst)
        for bit in itertools.chain(range(229, 241), range(201, 203)):
            # plut_addr and frame_rot metadata
            check_invalid(vi | (1 << bit), 'reserved')
            check_invalid(stm_vi | (1 << bit), 'reserved')

        for bit in itertools.chain(range(241, 245), range(224, 229), range(213, 216),
                                   range(200, 201)):
            check_invalid(vi | (1 << bit), 'reserved')
            check_invalid(plut_vi | (1 << bit), 'reserved')
            check_invalid(stm_vi | (1 << bit), 'reserved')
            check_invalid(stm_frm_vi | (1 << bit), 'reserved')

    metadatas = itertools.product(range(0, 8, 2), (0, 1), (False, True), (False, True),
                                  (False, True), range(4), range(4),
                                  (4, 1000000, 1099511627775))

    for chn, tone, trig, eof, clr, fwd, inv, cycles in metadatas:
        param_list = [f"frame_rot{tone}"]
        inst = Jaqal_v1_3.frame_pulse((0, 0, 0, 0), cycles, trig, eof, clr, fwd, inv)
        inst = Jaqal_v1_3.apply_channel_mask(inst, chn)
        inst = Jaqal_v1_3.apply_modtype_mask(inst, param_list)

        addr = random.randint(0, 4095)
        plut_inst = Jaqal_v1_3.program_PLUT(inst, addr)
        stm_inst = Jaqal_v1_3.stream(inst)

        assert inst.channels == [chn]
        assert plut_inst.channels == [chn]
        assert stm_inst.channels == [chn]

        vi = int(inst)

        assert inst == MatchPulse('pulse_data', param_list, [chn], cycles=cycles,
                                  trig=trig, eof=eof, clr=clr, fwd=fwd, inv=inv,
                                  ispl=(0, 0, 0, 0), shift=0, spl_phase=(0, 0, 0, 0))
        assert plut_inst == MatchPulse('plut', (), [chn], addr, cycles=cycles,
                                       trig=trig, eof=eof, clr=clr, fwd=fwd, inv=inv,
                                       ispl=(0, 0, 0, 0), shift=0, spl_phase=(0, 0, 0, 0))
        assert stm_inst == MatchPulse('stream', param_list, [chn], cycles=cycles,
                                      trig=trig, eof=eof, clr=clr, fwd=fwd, inv=inv,
                                      ispl=(0, 0, 0, 0), shift=0, spl_phase=(0, 0, 0, 0))
        assert Jaqal_v1_3.dump_insts(inst.to_bytes()) == f'invalid(reserved): {vi:0>64x}'
        assert Jaqal_v1_3.dump_insts(plut_inst.to_bytes()) == str(plut_inst)
        assert Jaqal_v1_3.dump_insts(stm_inst.to_bytes()) == str(stm_inst)
        assert Jaqal_v1_3.dump_insts(inst.to_bytes() + stm_inst.to_bytes() + plut_inst.to_bytes()) == f'invalid(reserved): {vi:0>64x}' + '\n' + str(stm_inst) + '\n' + str(plut_inst)

        plut_vi = int(plut_inst)
        stm_vi = int(stm_inst)
        for bit in range(229, 241):
            # plut_addr
            check_invalid(vi | (1 << bit), 'reserved')
            check_invalid(stm_vi | (1 << bit), 'reserved')

        for bit in itertools.chain(range(241, 245), range(224, 229),
                                   range(213, 216), range(200, 201)):
            check_invalid(vi | (1 << bit), 'reserved')
            check_invalid(plut_vi | (1 << bit), 'reserved')
            check_invalid(stm_vi | (1 << bit), 'reserved')

def test_freq_spline():
    def freq_pulse(spline, cycles):
        inst = Jaqal_v1_3.freq_pulse(spline, cycles, False, False, False)
        inst = Jaqal_v1_3.apply_channel_mask(inst, 0)
        return Jaqal_v1_3.apply_modtype_mask(inst, 'freq0')
    def match(fspl=None, **kwargs):
        return MatchPulse('pulse_data', ['freq0'], [0], trig=False, sync=False,
                          enable=False, ff=False, spl_freq=fspl, **kwargs)
    inst = freq_pulse((-409.6e6, 0, 0, 0), 1000)
    assert inst == match(cycles=1000, shift=0, ispl=(-0x8000000000,), fspl=(-409.6e6,))

    inst = freq_pulse((204.8e6, 0, 0, 0), 2100)
    assert inst == match(cycles=2100, shift=0, ispl=(0x4000000000,), fspl=(204.8e6,))

    inst = freq_pulse((-204.8e6, 819.2e6, 0, 0), 4)
    assert inst == match(cycles=4, shift=0, ispl=(-0x4000000000, 0x4000000000),
                         fspl=(-204.8e6, 819.2e6))

    inst = freq_pulse((-204.8e6, 819.2e6 * 2, 0, 0), 4)
    assert inst == match(cycles=4, shift=0, ispl=(-0x4000000000, -0x8000000000),
                         fspl=(-204.8e6, -1638.4e6))

    inst = freq_pulse((-204.8e6, 8.191999999991808e8, 0, 0), 4)
    assert inst == match(cycles=4, shift=0, ispl=(-0x4000000000, 0x4000000000),
                         fspl=(-204.8e6, 819.2e6))

    # Test the exact rounding threshold to make sure we are rounding things correctly.
    inst = freq_pulse((-204.8e6, 8.191999999985099e8, 0, 0), 4)
    assert inst == match(cycles=4, shift=0, ispl=(-0x4000000000, 0x4000000000),
                         fspl=(-204.8e6, 819.2e6))

    inst = freq_pulse((-204.8e6, 8.191999999985098e8, 0, 0), 4)
    assert inst == match(cycles=4, shift=1, ispl=(-0x4000000000, 0x7fffffffff),
                         fspl=(-204.8e6, 819199999.9985099))

    # Higher orders
    inst = freq_pulse((204.8e6, 0, 819.2e6, 0), 4)
    assert inst == match(cycles=4, fspl=(204.8e6, 0, 819.2e6), shift=0,
                         ispl=(0x4000000000, 0x1000000000, 0x2000000000))

    inst = freq_pulse((204.8e6, 0, 1638.4e6, 0), 4)
    assert inst == match(cycles=4, fspl=(204.8e6, 0, 1638.4e6), shift=0,
                         ispl=(0x4000000000, 0x2000000000, 0x4000000000))

    inst = freq_pulse((204.8e6, 0, 409.6e6, 0), 4)
    assert inst == match(cycles=4, fspl=(204.8e6, 0, 409.6e6), shift=1,
                         ispl=(0x4000000000, 0x1000000000, 0x4000000000))

    inst = freq_pulse((204.8e6, 0, 0, 819.2e6), 4)
    assert inst == match(cycles=4, fspl=(204.8e6, 0, 0, 819.2e6), shift=0,
                         ispl=(0x4000000000, 0x400000000, 0x1800000000, 0x1800000000))

    inst = freq_pulse((204.8e6, 0, 0, 409.6e6), 4)
    assert inst == match(cycles=4, fspl=(204.8e6, 0, 0, 409.6e6), shift=1,
                         ispl=(0x4000000000, 0x400000000, 0x3000000000, 0x6000000000))

    inst = freq_pulse((204.8e6, 0, 0, 409.6e6), 4096)
    assert inst == match(cycles=4096, fspl=(204.8e6, 0, 0, 409.6e6), shift=11,
                         ispl=(0x4000000000, 0x4000, 0xc000000, 0x6000000000))

    for i in range(2000):
        o0 = random.random() * 100e6
        o1 = random.random() * 200e6 - 100e6
        o2 = random.random() * 200e6 - 100e6
        o3 = random.random() * 200e6 - 100e6
        cycles = random.randint(4, 1000000)
        inst = freq_pulse((o0, o1, o2, o3), cycles)
        assert inst == match(cycles=cycles, fspl=(o0, o1, o2, o3), abs=0.05, rel=1e-10)

def test_amp_spline():
    def amp_pulse(spline, cycles):
        inst = Jaqal_v1_3.amp_pulse(spline, cycles, False, False, False)
        inst = Jaqal_v1_3.apply_channel_mask(inst, 0)
        return Jaqal_v1_3.apply_modtype_mask(inst, 'amp0')
    def match(fspl=None, **kwargs):
        return MatchPulse('pulse_data', ['amp0'], [0], trig=False, sync=False,
                          enable=False, ff=False, spl_amp=fspl, **kwargs)
    inst = amp_pulse((1, 0, 0, 0), 1000)
    assert inst == match(cycles=1000, shift=0, ispl=(0x7fff800000,), fspl=(1,))

    inst = amp_pulse((-1, 0, 0, 0), 2100)
    assert inst == match(cycles=2100, shift=0, ispl=(-0x7fff800000,), fspl=(-1,))

    inst = amp_pulse((-1, 2.0000305180437934, 0, 0), 4)
    assert inst == match(cycles=4, shift=0, ispl=(-0x7fff800000, 0x4000000000),
                         fspl=(-1, 2.0000305180437934))

    # Test the exact rounding threshold to make sure we are rounding things correctly.
    inst = amp_pulse((-1, 2.0000000000000004, 0, 0), 4)
    assert inst == match(cycles=4, shift=0, ispl=(-0x7fff800000, 0x4000000000),
                         fspl=(-1, 2.0000305180437934))

    inst = amp_pulse((-1, 2, 0, 0), 4)
    assert inst == match(cycles=4, shift=1, ispl=(-0x7fff800000, 0x7fff800000),
                         fspl=(-1, 2))

    # Higher orders
    inst = amp_pulse((-1, 0, 2.0000305180437934, 0), 4)
    assert inst == match(cycles=4, fspl=(-1, 0, 2.0000305180437934), shift=0,
                         ispl=(-0x7fff800000, 0x1000000000, 0x2000000000))

    inst = amp_pulse((-1, 0, 2.0000305180437934, 0), 1024)
    assert inst == match(cycles=1024, fspl=(-1, 0, 2.0000305180437934), shift=8,
                         ispl=(-0x7fff800000, 0x10000000, 0x2000000000))

    inst = amp_pulse((1, 0, -2.0000305180437934, 0), 1024)
    assert inst == match(cycles=1024, fspl=(1, 0, -2.0000305180437934), shift=8,
                         ispl=(0x7fff800000, -0x10000000, -0x2000000000))

    inst = amp_pulse((-1, 0, 0, 2.0000305180437934), 4)
    assert inst == match(cycles=4, fspl=(-1, 0, 0, 2.0000305180437934), shift=0,
                         ispl=(-0x7fff800000, 0x400000000, 0x1800000000, 0x1800000000))

    inst = amp_pulse((-1, 0, 0, 2.0000305180437934), 2048)
    assert inst == match(cycles=2048, shift=9,
                         ispl=(-0x7fff800000, 0, 0xc000000, 0x1800000000),
                         fspl=(-1, 0, 0, 2.0000305180437934), abs=5e-7)

    for i in range(2000):
        o0 = random.random() * 2 - 1
        o1 = random.random() * 2 - 1
        o2 = random.random() * 2 - 1
        o3 = random.random() * 2 - 1
        cycles = random.randint(4, 1000000)
        inst = amp_pulse((o0, o1, o2, o3), cycles)
        assert inst == match(cycles=cycles, fspl=(o0, o1, o2, o3), abs=1e-4, rel=1e-4)

class PhaseTester:
    @staticmethod
    def pulse(spline, cycles):
        inst = Jaqal_v1_3.phase_pulse(spline, cycles, False, False, False)
        inst = Jaqal_v1_3.apply_channel_mask(inst, 0)
        return Jaqal_v1_3.apply_modtype_mask(inst, 'phase0')

    @staticmethod
    def match(fspl=None, **kws):
        return MatchPulse('pulse_data', ['phase0'], [0], trig=False, sync=False,
                          enable=False, ff=False, spl_phase=fspl, **kws)

class FrameTester:
    @staticmethod
    def pulse(spline, cycles):
        inst = Jaqal_v1_3.frame_pulse(spline, cycles, False, False, False, 0, 0)
        inst = Jaqal_v1_3.apply_channel_mask(inst, 0)
        return Jaqal_v1_3.apply_modtype_mask(inst, 'frame_rot0')

    @staticmethod
    def match(fspl=None, **kws):
        return MatchPulse('pulse_data', ['frame_rot0'], [0], trig=False, eof=False,
                          clr=False, fwd=0, inv=0, spl_phase=fspl, **kws)

@pytest.mark.parametrize('cls', [PhaseTester, FrameTester])
def test_phase_spline(cls):
    inst = cls.pulse((-0.5, 0, 0, 0), 1000)
    assert inst == cls.match(cycles=1000, shift=0, ispl=(-0x8000000000,), fspl=(-0.5,))

    inst = cls.pulse((0.25, 0, 0, 0), 2100)
    assert inst == cls.match(cycles=2100, shift=0, ispl=(0x4000000000,), fspl=(0.25,))

    inst = cls.pulse((-0.25, 1, 0, 0), 4)
    assert inst == cls.match(cycles=4, fspl=(-0.25, 1), shift=0,
                             ispl=(-0x4000000000, 0x4000000000))

    inst = cls.pulse((-0.25, 2, 0, 0), 4)
    assert inst == cls.match(cycles=4, fspl=(-0.25, -2), shift=0,
                             ispl=(-0x4000000000, -0x8000000000))

    inst = cls.pulse((-0.25, 0.999999999999, 0, 0), 4)
    assert inst == cls.match(cycles=4, fspl=(-0.25, 1), shift=0,
                             ispl=(-0x4000000000, 0x4000000000))

    # Test the exact rounding threshold to make sure we are rounding things correctly.
    inst = cls.pulse((-0.25, 0.9999999999981811, 0, 0), 4)
    assert inst == cls.match(cycles=4, fspl=(-0.25, 1), shift=0,
                             ispl=(-0x4000000000, 0x4000000000))

    inst = cls.pulse((-0.25, 0.999999999998181, 0, 0), 4)
    assert inst == cls.match(cycles=4, fspl=(-0.25, 0.999999999998181), shift=1,
                             ispl=(-0x4000000000, 0x7fffffffff))

    # Higher orders
    inst = cls.pulse((0.25, 0, 1, 0), 4)
    assert inst == cls.match(cycles=4, fspl=(0.25, 0, 1), shift=0,
                             ispl=(0x4000000000, 0x1000000000, 0x2000000000))

    inst = cls.pulse((0.25, 0, 2, 0), 4)
    assert inst == cls.match(cycles=4, fspl=(0.25, 0, 2), shift=0,
                             ispl=(0x4000000000, 0x2000000000, 0x4000000000))

    inst = cls.pulse((0.25, 0, 0.5, 0), 4)
    assert inst == cls.match(cycles=4, fspl=(0.25, 0, 0.5), shift=1,
                             ispl=(0x4000000000, 0x1000000000, 0x4000000000))

    inst = cls.pulse((0.25, 0, 0, 1), 4)
    assert inst == cls.match(cycles=4, fspl=(0.25, 0, 0, 1), shift=0,
                             ispl=(0x4000000000, 0x400000000,
                                   0x1800000000, 0x1800000000))

    inst = cls.pulse((0.25, 0, 0, 0.5), 4)
    assert inst == cls.match(cycles=4, fspl=(0.25, 0, 0, 0.5), shift=1,
                             ispl=(0x4000000000, 0x400000000,
                                   0x3000000000, 0x6000000000))

    inst = cls.pulse((0.25, 0, 0, 0.5), 4096)
    assert inst == cls.match(cycles=4096, fspl=(0.25, 0, 0, 0.5), shift=11,
                             ispl=(0x4000000000, 0x4000, 0xc000000, 0x6000000000))

    for i in range(2000):
        o0 = random.random() * 0.99 - 0.5
        o1 = random.random() * 0.99 - 0.5
        o2 = random.random() * 0.99 - 0.5
        o3 = random.random() * 0.99 - 0.5
        cycles = random.randint(4, 1000000)
        inst = cls.pulse((o0, o1, o2, o3), cycles)
        assert inst == cls.match(cycles=cycles, fspl=(o0, o1, o2, o3),
                                 abs=1e-10, rel=1e-10)

def rand_pulse(chn):
    tone = random.randint(0, 1)
    cycles = random.randint(10, 11)
    ty = random.randint(0, 3)
    o0 = random.randint(0, 3) / 3
    o1 = random.randint(0, 3) / 3
    o2 = random.randint(0, 3) / 3
    o3 = random.randint(0, 3) / 3
    if ty == 0:
        pulse = Jaqal_v1_3.freq_pulse((o0 * 100e6, o1 * 200e6 - 100e6,
                                      o2 * 200e6 - 100e6, o3 * 200e6 - 100e6),
                                      cycles, False, False, False)
        pulse = Jaqal_v1_3.apply_modtype_mask(pulse, f'freq{tone}')
    elif ty == 1:
        pulse = Jaqal_v1_3.amp_pulse((o0 * 2 - 1, o1 * 2 - 1, o2 * 2 - 1, o3 * 2 - 1),
                                     cycles, False, False, False)
        pulse = Jaqal_v1_3.apply_modtype_mask(pulse, f'amp{tone}')
    elif ty == 2:
        pulse = Jaqal_v1_3.phase_pulse((o0 * 2 - 1, o1 * 2 - 1, o2 * 2 - 1, o3 * 2 - 1),
                                       cycles, False, False, False)
        pulse = Jaqal_v1_3.apply_modtype_mask(pulse, f'phase{tone}')
    else:
        pulse = Jaqal_v1_3.frame_pulse((o0 * 2 - 1, o1 * 2 - 1, o2 * 2 - 1, o3 * 2 - 1),
                                       cycles, False, False, False, 0, 0)
        pulse = Jaqal_v1_3.apply_modtype_mask(pulse, f'frame_rot{tone}')
    return Jaqal_v1_3.apply_channel_mask(pulse, chn)

def test_extract_pulse():
    pulses = [rand_pulse(random.randint(0, 7)) for _ in range(3000)]
    stream_bytes = b''
    for pulse in pulses:
        stream_bytes += Jaqal_v1_3.stream(pulse).to_bytes()
    assert Jaqal_v1_3.extract_pulses(stream_bytes) == pulses

    gseq = Jaqal_v1_3.sequence(2, 0, [0, 1, 2, 3]).to_bytes()
    wait_anc = Jaqal_v1_3.sequence(2, 1, [0, 1, 2, 3]).to_bytes()
    cont_anc = Jaqal_v1_3.sequence(2, 2, [0, 1, 2, 3]).to_bytes()
    glut = Jaqal_v1_3.program_GLUT(2, [0, 1, 2, 3], [0, 3, 6, 10],
                                   [2, 5, 9, 10]).to_bytes()
    glut_inverse = Jaqal_v1_3.program_GLUT(2, [0, 1, 2, 3],
                                           [2, 5, 9, 10], [0, 3, 6, 10]).to_bytes()

    plut_mem = [rand_pulse(2) for _ in range(4)]
    pulse_modtpes = [inst.to_dict()['modtype'] for inst in plut_mem]

    slut1 = Jaqal_v1_3.program_SLUT(2, [0, 1, 2, 3, 4, 5],
                                    [0, 1, 2, 3, 0, 1],
                                    [*pulse_modtpes, pulse_modtpes[0],
                                     pulse_modtpes[1]]).to_bytes()
    slut2 = Jaqal_v1_3.program_SLUT(2, [6, 7, 8, 9, 10, 11], [2, 3, 0, 1, 2, 3],
                                    [pulse_modtpes[2], pulse_modtpes[3],
                                    *pulse_modtpes]).to_bytes()

    plut = b''.join(Jaqal_v1_3.program_PLUT(plut_mem[i], i).to_bytes() for i in range(4))

    invalid_pulse_base = Jaqal_v1_3.apply_channel_mask(invalid_pulse, 2)
    invalid_plut = [Jaqal_v1_3.apply_modtype_mask(invalid_pulse_base, modtype)
                    for modtype in pulse_modtpes]

    # Out of bound gate
    assert Jaqal_v1_3.extract_pulses(gseq) == [invalid_pulse] * 4
    # Out of bound sequence
    assert Jaqal_v1_3.extract_pulses(glut + gseq) == [invalid_pulse] * 11
    assert Jaqal_v1_3.extract_pulses(glut_inverse + gseq) == [invalid_pulse] * 4
    # Out of bound pulse
    assert Jaqal_v1_3.extract_pulses(slut1 + slut2 + glut + gseq) == invalid_plut * 2 + invalid_plut[:3]

    assert Jaqal_v1_3.extract_pulses(plut + slut1 + slut2 + glut + gseq) == plut_mem * 2 + plut_mem[:3]
    assert Jaqal_v1_3.extract_pulses(plut + slut1 + slut2 + glut + wait_anc) == [invalid_pulse]
    assert Jaqal_v1_3.extract_pulses(plut + slut1 + slut2 + glut + cont_anc) == [invalid_pulse]

def test_extract_pulse_broadcast():
    def apply_masks(inst, modtype, chns):
        return Jaqal_v1_3.apply_channel_mask(
            Jaqal_v1_3.apply_modtype_mask(inst, modtype), chns)

    pulse = Jaqal_v1_3.amp_pulse((0.5, 0.1, 0.3, -0.5), 200, False, False, False)
    pulse_mod = Jaqal_v1_3.apply_modtype_mask(pulse, ('amp0', 'freq1', 'phase0'))
    pulse_chn = Jaqal_v1_3.apply_channel_mask(pulse_mod, (0, 1, 5))
    stream_inst = Jaqal_v1_3.stream(pulse_chn)
    assert Jaqal_v1_3.extract_pulses(stream_inst.to_bytes(), False) == [
        Jaqal_v1_3.apply_channel_mask(pulse_mod, 0),
        Jaqal_v1_3.apply_channel_mask(pulse_mod, 1),
        Jaqal_v1_3.apply_channel_mask(pulse_mod, 5)
    ]
    assert Jaqal_v1_3.extract_pulses(stream_inst.to_bytes()) == [
        apply_masks(pulse, 'amp0', 0),
        apply_masks(pulse, 'phase0', 0),
        apply_masks(pulse, 'freq1', 0),
        apply_masks(pulse, 'amp0', 1),
        apply_masks(pulse, 'phase0', 1),
        apply_masks(pulse, 'freq1', 1),
        apply_masks(pulse, 'amp0', 5),
        apply_masks(pulse, 'phase0', 5),
        apply_masks(pulse, 'freq1', 5),
    ]
    assert Jaqal_v1_3.extract_pulses(
        Jaqal_v1_3.apply_modtype_mask(stream_inst, ()).to_bytes(), False) == [
        apply_masks(pulse, (), 0), apply_masks(pulse, (), 1), apply_masks(pulse, (), 5)
    ]
    assert Jaqal_v1_3.extract_pulses(
        Jaqal_v1_3.apply_modtype_mask(stream_inst, ()).to_bytes()) == []

    pulse1 = Jaqal_v1_3.amp_pulse((0.5, 0.1, 0.3, -0.5), 200, False, False, False)
    pulse2 = Jaqal_v1_3.amp_pulse((0.5, 0.2, 0.2, -0.1), 300, False, False, False)
    pulse3 = Jaqal_v1_3.amp_pulse((0.1, 0.3, 0.1, -0.3), 400, False, False, False)
    pulse4 = Jaqal_v1_3.amp_pulse((0.2, 0.4, -0.3, 0.5), 600, False, False, False)

    insts = []
    insts.append(Jaqal_v1_3.program_PLUT(apply_masks(pulse1, ('amp0', 'freq1'),
                                                     (0, 1)), 0))
    insts.append(Jaqal_v1_3.program_PLUT(apply_masks(pulse2, (), (0)), 1))
    insts.append(Jaqal_v1_3.program_PLUT(apply_masks(pulse3, ('phase0',), (1)), 1))
    insts.append(Jaqal_v1_3.program_PLUT(apply_masks(pulse4, (), (1, 0)), 2))
    insts.append(Jaqal_v1_3.program_SLUT((0, 1), [0, 1, 2], [0, 1, 2],
                                         ['amp0', ['amp0', 'amp1', 'phase1'],
                                          ('amp1',)]))
    insts.append(Jaqal_v1_3.program_GLUT((0, 1), [0], [0], [2]))
    insts.append(Jaqal_v1_3.sequence((0, 1), 0, [0]))
    assert Jaqal_v1_3.extract_pulses(b''.join(inst.to_bytes() for inst in insts)) == [
        apply_masks(pulse1, 'amp0', 0),
        apply_masks(pulse2, 'amp0', 0),
        apply_masks(pulse2, 'amp1', 0),
        apply_masks(pulse2, 'phase1', 0),
        apply_masks(pulse4, 'amp1', 0),
        apply_masks(pulse1, 'amp0', 1),
        apply_masks(pulse3, 'amp0', 1),
        apply_masks(pulse3, 'amp1', 1),
        apply_masks(pulse3, 'phase1', 1),
        apply_masks(pulse4, 'amp1', 1),
    ]
    assert Jaqal_v1_3.extract_pulses(b''.join(inst.to_bytes() for inst in insts), False) == [
        apply_masks(pulse1, 'amp0', 0),
        apply_masks(pulse2, ('amp0', 'amp1', 'phase1'), 0),
        apply_masks(pulse4, 'amp1', 0),
        apply_masks(pulse1, 'amp0', 1),
        apply_masks(pulse3, ('amp0', 'amp1', 'phase1'), 1),
        apply_masks(pulse4, 'amp1', 1),
    ]

    insts = []
    insts.append(Jaqal_v1_3.program_PLUT(apply_masks(pulse1, ('amp0', 'freq1'),
                                                     (0, 1)), 0))
    insts.append(Jaqal_v1_3.program_PLUT(apply_masks(pulse2, (), (0)), 1))
    insts.append(Jaqal_v1_3.program_PLUT(apply_masks(pulse3, ('phase0',), (1)), 1))
    insts.append(Jaqal_v1_3.program_PLUT(apply_masks(pulse4, (), (1, 0)), 2))
    insts.append(Jaqal_v1_3.program_SLUT(0, [0, 1, 2], [0, 1, 2],
                                         ['amp0', ['amp0', 'amp1', 'phase1'],
                                          ('amp1',)]))
    insts.append(Jaqal_v1_3.program_SLUT(1, [0, 1, 2], [2, 1, 0],
                                         [['freq0', 'phase1'], ['amp0', 'amp1'],
                                          ('amp1', 'frame_rot0')]))
    insts.append(Jaqal_v1_3.program_GLUT((0, 1), [0], [0], [2]))
    insts.append(Jaqal_v1_3.sequence((0, 1), 0, [0]))
    assert Jaqal_v1_3.extract_pulses(b''.join(inst.to_bytes() for inst in insts)) == [
        apply_masks(pulse1, 'amp0', 0),
        apply_masks(pulse2, 'amp0', 0),
        apply_masks(pulse2, 'amp1', 0),
        apply_masks(pulse2, 'phase1', 0),
        apply_masks(pulse4, 'amp1', 0),
        apply_masks(pulse4, 'freq0', 1),
        apply_masks(pulse4, 'phase1', 1),
        apply_masks(pulse3, 'amp0', 1),
        apply_masks(pulse3, 'amp1', 1),
        apply_masks(pulse1, 'amp1', 1),
        apply_masks(pulse1, 'frame_rot0', 1),
    ]
    assert Jaqal_v1_3.extract_pulses(b''.join(inst.to_bytes() for inst in insts), False) == [
        apply_masks(pulse1, 'amp0', 0),
        apply_masks(pulse2, ('amp0', 'amp1', 'phase1'), 0),
        apply_masks(pulse4, 'amp1', 0),
        apply_masks(pulse4, ('freq0', 'phase1'), 1),
        apply_masks(pulse3, ('amp0', 'amp1'), 1),
        apply_masks(pulse1, ('amp1', 'frame_rot0'), 1),
    ]

    insts = []
    insts.append(Jaqal_v1_3.program_PLUT(apply_masks(pulse1, ('amp0', 'freq1'),
                                                     (0, 1)), 0))
    insts.append(Jaqal_v1_3.program_PLUT(apply_masks(pulse2, (), (0)), 1))
    insts.append(Jaqal_v1_3.program_PLUT(apply_masks(pulse3, ('phase0',), (1)), 1))
    insts.append(Jaqal_v1_3.program_PLUT(apply_masks(pulse4, (), (1, 0)), 2))
    insts.append(Jaqal_v1_3.program_SLUT((0, 1), [0, 1, 2], [0, 1, 2],
                                         ['amp0', ['amp0', 'amp1', 'phase1'],
                                          ('amp1',)]))
    insts.append(Jaqal_v1_3.program_GLUT(0, [0], [0], [2]))
    insts.append(Jaqal_v1_3.program_GLUT(1, [0, 1], [0, 2], [1, 2]))
    insts.append(Jaqal_v1_3.sequence(1, 0, [1]))
    insts.append(Jaqal_v1_3.sequence((0, 1), 0, [0]))
    assert Jaqal_v1_3.extract_pulses(b''.join(inst.to_bytes() for inst in insts)) == [
        apply_masks(pulse4, 'amp1', 1),
        apply_masks(pulse1, 'amp0', 0),
        apply_masks(pulse2, 'amp0', 0),
        apply_masks(pulse2, 'amp1', 0),
        apply_masks(pulse2, 'phase1', 0),
        apply_masks(pulse4, 'amp1', 0),
        apply_masks(pulse1, 'amp0', 1),
        apply_masks(pulse3, 'amp0', 1),
        apply_masks(pulse3, 'amp1', 1),
        apply_masks(pulse3, 'phase1', 1),
    ]
    assert Jaqal_v1_3.extract_pulses(b''.join(inst.to_bytes() for inst in insts), False) == [
        apply_masks(pulse4, 'amp1', 1),
        apply_masks(pulse1, 'amp0', 0),
        apply_masks(pulse2, ('amp0', 'amp1', 'phase1'), 0),
        apply_masks(pulse4, 'amp1', 0),
        apply_masks(pulse1, 'amp0', 1),
        apply_masks(pulse3, ('amp0', 'amp1', 'phase1'), 1),
    ]
