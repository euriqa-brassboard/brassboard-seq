#

from brassboard_seq.rfsoc_backend import Jaqal_v1, JaqalInst_v1, JaqalChannelGen_v1

import pytest
import re
import random
import itertools

class MatchGLUT:
    def __init__(self, chn=None, gaddrs=None, starts=None, ends=None, cnt=None):
        self.chn = chn
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
        if not isinstance(other, JaqalInst_v1):
            return NotImplemented
        d = other.to_dict()
        assert d['type'] == 'glut'
        lut_str = ''.join(f' [{gaddr}]=[{start},{end}]' for gaddr, start, end in
                            zip(d['gaddrs'], d['starts'], d['ends']))
        assert str(other) == f"glut.{d['channel']}[{d['count']}]{lut_str}"
        if self.chn is not None:
            assert d['channel'] == self.chn
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
    def __init__(self, chn=None, saddrs=None, paddrs=None, cnt=None):
        self.chn = chn
        self.saddrs = list(saddrs) if saddrs is not None else None
        self.paddrs = list(paddrs) if paddrs is not None else None
        if self.saddrs is not None:
            assert cnt is None or cnt == len(self.saddrs)
            cnt = len(self.saddrs)
        if self.paddrs is not None:
            assert cnt is None or cnt == len(self.paddrs)
            cnt = len(self.paddrs)
        self.cnt = cnt

    def __eq__(self, other):
        if not isinstance(other, JaqalInst_v1):
            return NotImplemented
        d = other.to_dict()
        assert d['type'] == 'slut'
        lut_str = ''.join(f' [{saddr}]={paddr}' for saddr, paddr in
                            zip(d['saddrs'], d['paddrs']))
        assert str(other) == f"slut.{d['channel']}[{d['count']}]{lut_str}"
        if self.chn is not None:
            assert d['channel'] == self.chn
        if self.cnt is not None:
            assert d['count'] == self.cnt
        if self.saddrs is not None:
            assert d['saddrs'] == self.saddrs
        if self.paddrs is not None:
            assert d['paddrs'] == self.paddrs
        return True

class MatchGSEQ:
    def __init__(self, chn=None, mode=None, gaddrs=None, cnt=None):
        self.chn = chn
        self.gaddrs = list(gaddrs) if gaddrs is not None else None
        self.mode = mode
        if self.gaddrs is not None:
            assert cnt is None or cnt == len(self.gaddrs)
            cnt = len(self.gaddrs)
        self.cnt = cnt

    def __eq__(self, other):
        if not isinstance(other, JaqalInst_v1):
            return NotImplemented
        d = other.to_dict()
        seq_str = ''.join(f' {gaddr}' for gaddr in d['gaddrs'])
        assert str(other) == f"{d['type']}.{d['channel']}[{d['count']}]{seq_str}"
        if self.mode is not None:
            assert d['type'] == self.mode
        if self.chn is not None:
            assert d['channel'] == self.chn
        if self.cnt is not None:
            assert d['count'] == self.cnt
        if self.gaddrs is not None:
            assert d['gaddrs'] == self.gaddrs
        return True

def pad_list(lst, n, v):
    return lst + [v] * (n - len(lst))

class MatchPulse:
    def __init__(self, param, chn, tone, mode, addr, cycles,
                 ispl, shift, fspl, _rel, _abs, approx):
        self.chn = chn
        self.tone = tone
        self.mode = mode
        self.addr = addr
        if addr is not None:
            assert self.mode is None or self.mode == 'plut'
            self.mode = 'plut'
        assert param is not None
        self.param = param
        self.cycles = cycles
        if fspl is None:
            assert _rel is None
            assert _abs is None
        self.ispl = ispl
        self.shift = shift
        self.fspl = fspl
        self.rel = _rel
        self.abs = _abs
        self.approx = approx or _rel is not None or _abs is not None

    def check_prefix(self, other):
        assert isinstance(other, JaqalInst_v1)
        d = other.to_dict()

        str_inst = str(other)
        repr_inst = repr(other)
        m_str = re.match('^(.*) {([^{}]+)}(.*)$', str_inst)
        assert m_str is not None
        m_repr = re.match('^(.*) {([^{}]+)}(.*)$', repr_inst)
        assert m_repr is not None

        plut_str = f"[{d['paddr']}]=" if d['type'] == 'plut' else ''
        assert m_str[1] == m_repr[1]
        assert m_str[1] == f"{d['type']}.{d['channel']} {plut_str}{d['param']}{d['tone']} <{d['cycles']}>"

        if self.mode is not None:
            assert d['type'] == self.mode
        if self.chn is not None:
            assert d['channel'] == self.chn
        if self.addr is not None:
            assert self.mode == 'plut'
            assert d['paddr'] == self.addr
        assert d['param'] == self.param
        if self.tone is not None:
            assert d['tone'] == self.tone
        if self.cycles is not None:
            assert d['cycles'] == self.cycles

        assert d['spline'] == pad_list([float(o) for o in m_str[2].split(', ')], 4, 0)
        if self.fspl is not None:
            expected = pad_list(list(self.fspl), 4, 0)
            if self.approx:
                expected = pytest.approx(expected, abs=self.abs, rel=self.rel)
            assert d['spline'] == expected

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

        if self.ispl is not None:
            assert spline_mu == pad_list(list(self.ispl), 4, 0)
        if self.shift is not None:
            assert spline_shift == self.shift

        flags = ' trig' if d['trig'] else ''
        if self.param == 'frame_rot':
            if d['eof']:
                flags += ' eof'
            if d['clr']:
                flags += ' clr'
            flags += f" fwd:{d['fwd']} inv:{d['inv']}"
        else:
            if d['sync']:
                flags += ' sync'
            if d['enable']:
                flags += ' enable'
            if d['ff']:
                flags += ' ff'

        assert m_str[3] == flags
        assert m_repr[3] == m_str[3]

        return d

class MatchParamPulse(MatchPulse):
    def __init__(self, param, chn=None, tone=None, mode=None, addr=None,
                 cycles=None, trig=None, sync=None, enable=None, ff=None,
                 ispl=None, shift=None, fspl=None, rel=None, abs=None, approx=None):
        assert param in ('freq', 'amp', 'phase')
        super().__init__(param, chn, tone, mode, addr, cycles,
                         ispl, shift, fspl, rel, abs, approx)
        self.trig = trig
        self.sync = sync
        self.enable = enable
        self.ff = ff

    def __eq__(self, other):
        if not isinstance(other, JaqalInst_v1):
            return NotImplemented
        flags = self.check_prefix(other)
        if self.trig is not None:
            assert flags['trig'] == self.trig
        if self.sync is not None:
            assert flags['sync'] == self.sync
        if self.enable is not None:
            assert flags['enable'] == self.enable
        if self.ff is not None:
            assert flags['ff'] == self.ff
        return True

class MatchFramePulse(MatchPulse):
    def __init__(self, chn=None, tone=None, mode=None, addr=None,
                 cycles=None, trig=None, eof=None, clr=None, fwd=None, inv=None,
                 ispl=None, shift=None, fspl=None, rel=None, abs=None, approx=None):
        super().__init__('frame_rot', chn, tone, mode, addr, cycles,
                         ispl, shift, fspl, rel, abs, approx)
        self.trig = trig
        self.eof = eof
        self.clr = clr
        self.fwd = fwd
        self.inv = inv

    def __eq__(self, other):
        if not isinstance(other, JaqalInst_v1):
            return NotImplemented
        flags = self.check_prefix(other)
        if self.trig is not None:
            assert flags['trig'] == self.trig
        if self.eof is not None:
            assert flags['eof'] == self.eof
        if self.clr is not None:
            assert flags['clr'] == self.clr
        if self.fwd is not None:
            assert flags['fwd'] == self.fwd
        if self.inv is not None:
            assert flags['inv'] == self.inv
        return True

invalid_pulse = JaqalInst_v1(b'\xff' * 32)

def check_invalid(v, msg):
    inst_bytes = v.to_bytes(32, 'little')
    assert Jaqal_v1.dump_insts(inst_bytes) == f'invalid({msg}): {v:0>64x}'
    d = JaqalInst_v1(inst_bytes).to_dict()
    assert d['type'] == 'invalid'
    assert d['error'] == msg
    assert d['inst'] == f'{v:0>64x}'
    pulses, = Jaqal_v1.extract_pulses(inst_bytes)
    assert pulses == invalid_pulse


def test_insts():
    assert Jaqal_v1.dump_insts(b'') == ''
    with pytest.raises(ValueError, match="Instruction stream length not a multiple of instruction size"):
        Jaqal_v1.dump_insts(b'x')

    assert str(JaqalInst_v1()) == 'pulse_data.0 freq0 <0> {0}'
    assert JaqalInst_v1().to_dict() == {'type': 'pulse_data', 'param': 'freq',
                                        'channel': 0, 'tone': 0, 'cycles': 0,
                                        'spline': [0.0, 0.0, 0.0, 0.0],
                                        'spline_mu': [0, 0, 0, 0], 'spline_shift': 0,
                                        'trig': False, 'sync': False,
                                        'enable': False, 'ff': False}
    with pytest.raises(TypeError, match=f"Invalid type '{int}'"):
        JaqalInst_v1(2)
    with pytest.raises(ValueError, match=f"Invalid address '-1'"):
        Jaqal_v1.program_PLUT(JaqalInst_v1(), -1)
    with pytest.raises(ValueError, match=f"Invalid address '4096'"):
        Jaqal_v1.program_PLUT(JaqalInst_v1(), 4096)

def test_glut():
    with pytest.raises(ValueError, match="Invalid channel number '-1'"):
        Jaqal_v1.program_GLUT(-1, [], [], [])
    with pytest.raises(ValueError, match="Invalid channel number '8'"):
        Jaqal_v1.program_GLUT(8, [], [], [])
    with pytest.raises(ValueError, match="Mismatch address length"):
        Jaqal_v1.program_GLUT(3, [1], [2, 3], [5])
    with pytest.raises(ValueError, match="Mismatch address length"):
        Jaqal_v1.program_GLUT(2, [1], [3], [5, 6])
    with pytest.raises(ValueError, match="Mismatch address length"):
        Jaqal_v1.program_GLUT(1, [1, 2], [3], [5])
    with pytest.raises(ValueError, match="Too many GLUT addresses to program"):
        Jaqal_v1.program_GLUT(0, list(range(7)), list(range(7)), list(range(7)))
    for n in range(7):
        for i in range(2000):
            chn = random.randint(0, 7)
            gaddrs = [random.randint(0, 4095) for _ in range(n)]
            starts = [random.randint(0, 4095) for _ in range(n)]
            ends = [random.randint(0, 4095) for _ in range(n)]
            inst = Jaqal_v1.program_GLUT(chn, gaddrs, starts, ends)
            assert inst.channel == chn
            assert inst == MatchGLUT(chn, gaddrs, starts, ends)
            assert str(inst) == repr(inst)
            assert Jaqal_v1.dump_insts(inst.to_bytes()) == str(inst)
            assert Jaqal_v1.dump_insts(inst.to_bytes() + inst.to_bytes()) == str(inst) + '\n' + str(inst)
            assert inst == JaqalInst_v1(inst.to_bytes())

    glut_base = 0x20000000000000000000000000000000000000000000000000000000000000
    for bit in itertools.chain(range(248, 256), range(239, 245), range(223, 229)):
        check_invalid(glut_base | (1 << bit), 'reserved')
    for n in range(7):
        glut_base_n = glut_base | (n << 236)
        for bit in range(n * 36, 220):
            check_invalid(glut_base_n | (1 << bit), 'reserved')
    for n in range(7, 8):
        check_invalid(glut_base | (n << 236), 'glut_oob')

def test_slut():
    with pytest.raises(ValueError, match="Invalid channel number '-1'"):
        Jaqal_v1.program_SLUT(-1, [], [])
    with pytest.raises(ValueError, match="Invalid channel number '8'"):
        Jaqal_v1.program_SLUT(8, [], [])
    with pytest.raises(ValueError, match="Mismatch address length"):
        Jaqal_v1.program_SLUT(3, [1], [2, 3])
    with pytest.raises(ValueError, match="Too many SLUT addresses to program"):
        Jaqal_v1.program_SLUT(0, list(range(10)), list(range(10)))
    for n in range(10):
        for i in range(3000):
            chn = random.randint(0, 7)
            saddrs = [random.randint(0, 4095) for _ in range(n)]
            paddrs = [random.randint(0, 4095) for _ in range(n)]
            inst = Jaqal_v1.program_SLUT(chn, saddrs, paddrs)
            assert inst.channel == chn
            assert inst == MatchSLUT(chn, saddrs, paddrs)
            assert str(inst) == repr(inst)
            assert Jaqal_v1.dump_insts(inst.to_bytes()) == str(inst)
            assert Jaqal_v1.dump_insts(inst.to_bytes() + inst.to_bytes()) == str(inst) + '\n' + str(inst)
            assert inst == JaqalInst_v1(inst.to_bytes())

    slut_base = 0x40000000000000000000000000000000000000000000000000000000000000
    for bit in itertools.chain(range(248, 256), range(240, 245), range(223, 229)):
        check_invalid(slut_base | (1 << bit), 'reserved')
    for n in range(10):
        slut_base_n = slut_base | (n << 236)
        for bit in range(n * 24, 220):
            check_invalid(slut_base_n | (1 << bit), 'reserved')
    for n in range(10, 16):
        check_invalid(slut_base | (n << 236), 'slut_oob')

def test_gseq():
    with pytest.raises(ValueError, match="Invalid channel number '-1'"):
        Jaqal_v1.sequence(-1, 0, [])
    with pytest.raises(ValueError, match="Invalid channel number '8'"):
        Jaqal_v1.sequence(8, 1, [])
    with pytest.raises(ValueError, match="Invalid sequencing mode 3"):
        Jaqal_v1.sequence(2, 3, [])
    with pytest.raises(ValueError, match="Too many GLUT addresses to sequence"):
        Jaqal_v1.sequence(3, 2, list(range(25)))
    for n in range(25):
        for i in range(2000):
            chn = random.randint(0, 7)
            mode = random.randint(0, 2)
            mode_str = ('gseq', 'wait_anc', 'cont_anc')[mode]
            gaddrs = [random.randint(0, 511) for _ in range(n)]
            inst = Jaqal_v1.sequence(chn, mode, gaddrs)
            assert inst.channel == chn
            assert inst == MatchGSEQ(chn, mode_str, gaddrs)
            assert str(inst) == repr(inst)
            assert Jaqal_v1.dump_insts(inst.to_bytes()) == str(inst)
            assert Jaqal_v1.dump_insts(inst.to_bytes() + inst.to_bytes()) == str(inst) + '\n' + str(inst)
            assert inst == JaqalInst_v1(inst.to_bytes())

    gseq_base = 0x80000000000000000000000000000000000000000000000000000000000000
    for bit in itertools.chain(range(248, 256), range(223, 239)):
        check_invalid(gseq_base | (1 << bit), 'reserved')
    for n in range(25):
        gseq_base_n = gseq_base | (n << 239)
        for bit in range(n * 9, 220):
            check_invalid(gseq_base_n | (1 << bit), 'reserved')
    for n in range(25, 64):
        check_invalid(gseq_base | (n << 239), 'gseq_oob')

def test_pulse_inst():
    with pytest.raises(ValueError, match="Invalid channel number '-1'"):
        Jaqal_v1.freq_pulse(-1, 0, (0, 0, 0, 0), 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid channel number '8'"):
        Jaqal_v1.freq_pulse(8, 1, (0, 0, 0, 0), 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid tone number '-1'"):
        Jaqal_v1.freq_pulse(3, -1, (0, 0, 0, 0), 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid tone number '2'"):
        Jaqal_v1.freq_pulse(4, 2, (0, 0, 0, 0), 100, False, False, False)
    with pytest.raises(TypeError, match=f"Invalid spline type '{int}'"):
        Jaqal_v1.freq_pulse(0, 1, 0, 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid cycle count '1099511627776'"):
        Jaqal_v1.freq_pulse(0, 1, (0, 0, 0, 0), 2**40, False, False, False)

    with pytest.raises(ValueError, match="Invalid channel number '-1'"):
        Jaqal_v1.amp_pulse(-1, 0, (0, 0, 0, 0), 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid channel number '8'"):
        Jaqal_v1.amp_pulse(8, 1, (0, 0, 0, 0), 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid tone number '-1'"):
        Jaqal_v1.amp_pulse(3, -1, (0, 0, 0, 0), 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid tone number '2'"):
        Jaqal_v1.amp_pulse(4, 2, (0, 0, 0, 0), 100, False, False, False)
    with pytest.raises(TypeError, match=f"Invalid spline type '{int}'"):
        Jaqal_v1.amp_pulse(0, 1, 0, 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid cycle count '1099511627776'"):
        Jaqal_v1.amp_pulse(0, 1, (0, 0, 0, 0), 2**40, False, False, False)

    with pytest.raises(ValueError, match="Invalid channel number '-1'"):
        Jaqal_v1.phase_pulse(-1, 0, (0, 0, 0, 0), 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid channel number '8'"):
        Jaqal_v1.phase_pulse(8, 1, (0, 0, 0, 0), 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid tone number '-1'"):
        Jaqal_v1.phase_pulse(3, -1, (0, 0, 0, 0), 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid tone number '2'"):
        Jaqal_v1.phase_pulse(4, 2, (0, 0, 0, 0), 100, False, False, False)
    with pytest.raises(TypeError, match=f"Invalid spline type '{int}'"):
        Jaqal_v1.phase_pulse(0, 1, 0, 100, False, False, False)
    with pytest.raises(ValueError, match="Invalid cycle count '1099511627776'"):
        Jaqal_v1.phase_pulse(0, 1, (0, 0, 0, 0), 2**40, False, False, False)

    with pytest.raises(ValueError, match="Invalid channel number '-1'"):
        Jaqal_v1.frame_pulse(-1, 0, (0, 0, 0, 0), 100, False, False, False, 0, 0)
    with pytest.raises(ValueError, match="Invalid channel number '8'"):
        Jaqal_v1.frame_pulse(8, 1, (0, 0, 0, 0), 100, False, False, False, 0, 0)
    with pytest.raises(ValueError, match="Invalid tone number '-1'"):
        Jaqal_v1.frame_pulse(3, -1, (0, 0, 0, 0), 100, False, False, False, 0, 0)
    with pytest.raises(ValueError, match="Invalid tone number '2'"):
        Jaqal_v1.frame_pulse(4, 2, (0, 0, 0, 0), 100, False, False, False, 0, 0)
    with pytest.raises(TypeError, match=f"Invalid spline type '{int}'"):
        Jaqal_v1.frame_pulse(0, 1, 0, 100, False, False, False, 0, 0)
    with pytest.raises(ValueError, match="Invalid cycle count '1099511627776'"):
        Jaqal_v1.frame_pulse(0, 1, (0, 0, 0, 0), 2**40, False, False, False, 0, 0)

    metadatas = itertools.product(('freq', 'amp', 'phase'), range(8), (0, 1),
                                  (False, True), (False, True), (False, True),
                                  (False, True), (4, 1000000, 1099511627775))

    for param, chn, tone, trig, sync, enable, ff, cycles in metadatas:
        flags = ''
        if trig:
            flags += ' trig'
        if sync:
            flags += ' sync'
        if enable:
            flags += ' enable'
        if ff:
            flags += ' ff'
        inst = getattr(Jaqal_v1, param + '_pulse')(chn, tone, (0, 0, 0, 0), cycles,
                                                   trig, sync, ff)
        vi = int(inst)
        assert vi & (1 << 226) == 0
        if enable:
            vi |= 1 << 226
            inst = JaqalInst_v1(vi.to_bytes(32, 'little'))

        addr = random.randint(0, 4095)
        plut_inst = Jaqal_v1.program_PLUT(inst, addr)
        stm_inst = Jaqal_v1.stream(inst)

        assert inst.channel == chn
        assert plut_inst.channel == chn
        assert stm_inst.channel == chn

        assert inst == MatchParamPulse(param, chn, tone, 'pulse_data', None, cycles,
                                       trig, sync, enable, ff, fspl=(0, 0, 0, 0),
                                       ispl=(0, 0, 0, 0), shift=31)
        assert plut_inst == MatchParamPulse(param, chn, tone, 'plut', addr, cycles,
                                            trig, sync, enable, ff, fspl=(0, 0, 0, 0),
                                            ispl=(0, 0, 0, 0), shift=31)
        assert stm_inst == MatchParamPulse(param, chn, tone, 'stream', None, cycles,
                                           trig, sync, enable, ff, fspl=(0, 0, 0, 0),
                                           ispl=(0, 0, 0, 0), shift=31)
        assert str(inst) == f'pulse_data.{chn} {param}{tone} <{cycles}> {{0}}{flags}'
        assert repr(inst) == f'pulse_data.{chn} {param}{tone} <{cycles}> {{0}}{flags}'
        assert Jaqal_v1.dump_insts(inst.to_bytes()) == f'invalid(reserved): {vi:0>64x}'

        assert str(plut_inst) == f'plut.{chn} [{addr}]={param}{tone} <{cycles}> {{0}}{flags}'
        assert repr(plut_inst) == f'plut.{chn} [{addr}]={param}{tone} <{cycles}> {{0}}{flags}'
        assert Jaqal_v1.dump_insts(plut_inst.to_bytes()) == str(plut_inst)

        assert str(stm_inst) == f'stream.{chn} {param}{tone} <{cycles}> {{0}}{flags}'
        assert repr(stm_inst) == f'stream.{chn} {param}{tone} <{cycles}> {{0}}{flags}'
        assert Jaqal_v1.dump_insts(stm_inst.to_bytes()) == str(stm_inst)
        assert Jaqal_v1.dump_insts(inst.to_bytes() + stm_inst.to_bytes() + plut_inst.to_bytes()) == f'invalid(reserved): {vi:0>64x}' + '\n' + str(stm_inst) + '\n' + str(plut_inst)

        plut_vi = int(plut_inst)
        stm_vi = int(stm_inst)
        for bit in range(229, 241):
            # plut_addr
            check_invalid(vi | (1 << bit), 'reserved')
            check_invalid(stm_vi | (1 << bit), 'reserved')

        for bit in itertools.chain(range(241, 245), range(228, 229), range(223, 224),
                                   range(200, 220)):
            check_invalid(vi | (1 << bit), 'reserved')
            check_invalid(plut_vi | (1 << bit), 'reserved')
            check_invalid(stm_vi | (1 << bit), 'reserved')

    metadatas = itertools.product(range(8), (0, 1), (False, True), (False, True),
                                  (False, True), range(4), range(4),
                                  (4, 1000000, 1099511627775))

    for chn, tone, trig, eof, clr, fwd, inv, cycles in metadatas:
        flags = ''
        if trig:
            flags += ' trig'
        if eof:
            flags += ' eof'
        if clr:
            flags += ' clr'
        flags += f' fwd:{fwd} inv:{inv}'
        inst = Jaqal_v1.frame_pulse(chn, tone, (0, 0, 0, 0), cycles,
                                    trig, eof, clr, fwd, inv)

        addr = random.randint(0, 4095)
        plut_inst = Jaqal_v1.program_PLUT(inst, addr)
        stm_inst = Jaqal_v1.stream(inst)

        assert inst.channel == chn
        assert plut_inst.channel == chn
        assert stm_inst.channel == chn

        vi = int(inst)

        assert inst == MatchFramePulse(chn, tone, 'pulse_data', None, cycles,
                                       trig, eof, clr, fwd, inv, fspl=(0, 0, 0, 0),
                                       ispl=(0, 0, 0, 0), shift=31)
        assert plut_inst == MatchFramePulse(chn, tone, 'plut', addr, cycles,
                                            trig, eof, clr, fwd, inv, fspl=(0, 0, 0, 0),
                                            ispl=(0, 0, 0, 0), shift=31)
        assert stm_inst == MatchFramePulse(chn, tone, 'stream', None, cycles,
                                           trig, eof, clr, fwd, inv, fspl=(0, 0, 0, 0),
                                           ispl=(0, 0, 0, 0), shift=31)
        assert str(inst) == f'pulse_data.{chn} frame_rot{tone} <{cycles}> {{0}}{flags}'
        assert repr(inst) == f'pulse_data.{chn} frame_rot{tone} <{cycles}> {{0}}{flags}'
        assert Jaqal_v1.dump_insts(inst.to_bytes()) == f'invalid(reserved): {vi:0>64x}'

        assert str(plut_inst) == f'plut.{chn} [{addr}]=frame_rot{tone} <{cycles}> {{0}}{flags}'
        assert repr(plut_inst) == f'plut.{chn} [{addr}]=frame_rot{tone} <{cycles}> {{0}}{flags}'
        assert Jaqal_v1.dump_insts(plut_inst.to_bytes()) == str(plut_inst)

        assert str(stm_inst) == f'stream.{chn} frame_rot{tone} <{cycles}> {{0}}{flags}'
        assert repr(stm_inst) == f'stream.{chn} frame_rot{tone} <{cycles}> {{0}}{flags}'
        assert Jaqal_v1.dump_insts(stm_inst.to_bytes()) == str(stm_inst)
        assert Jaqal_v1.dump_insts(inst.to_bytes() + stm_inst.to_bytes() + plut_inst.to_bytes()) == f'invalid(reserved): {vi:0>64x}' + '\n' + str(stm_inst) + '\n' + str(plut_inst)

        plut_vi = int(plut_inst)
        stm_vi = int(stm_inst)
        for bit in range(229, 241):
            # plut_addr
            check_invalid(vi | (1 << bit), 'reserved')
            check_invalid(stm_vi | (1 << bit), 'reserved')

        for bit in itertools.chain(range(241, 245), range(223, 224),
                                   range(200, 218)):
            check_invalid(vi | (1 << bit), 'reserved')
            check_invalid(plut_vi | (1 << bit), 'reserved')
            check_invalid(stm_vi | (1 << bit), 'reserved')

def test_freq_spline():
    inst = Jaqal_v1.freq_pulse(0, 0, (-409.6e6, 0, 0, 0), 1000, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=1000, shift=31,
                                   ispl=(-0x8000000000,), fspl=(-409.6e6,))

    inst = Jaqal_v1.freq_pulse(0, 0, (204.8e6, 0, 0, 0), 2100, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=2100, shift=31,
                                   ispl=(0x4000000000,), fspl=(204.8e6,))

    inst = Jaqal_v1.freq_pulse(0, 0, (-204.8e6, 819.2e6, 0, 0), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(-204.8e6, 819.2e6), shift=0,
                                   ispl=(-0x4000000000, 0x4000000000))

    inst = Jaqal_v1.freq_pulse(0, 0, (-204.8e6, 819.2e6 * 2, 0, 0), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(-204.8e6, -1638.4e6), shift=0,
                                   ispl=(-0x4000000000, -0x8000000000))

    inst = Jaqal_v1.freq_pulse(0, 0, (-204.8e6, 8.191999999991808e8, 0, 0), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(-204.8e6, 819.2e6), shift=0,
                                   ispl=(-0x4000000000, 0x4000000000))

    # Test the exact rounding threshold to make sure we are rounding things correctly.
    inst = Jaqal_v1.freq_pulse(0, 0, (-204.8e6, 8.191999999985099e8, 0, 0), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(-204.8e6, 819.2e6), shift=0,
                                   ispl=(-0x4000000000, 0x4000000000))

    inst = Jaqal_v1.freq_pulse(0, 0, (-204.8e6, 8.191999999985098e8, 0, 0), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(-204.8e6, 819199999.9985099),
                                   shift=1, ispl=(-0x4000000000, 0x7fffffffff))

    # Higher orders
    inst = Jaqal_v1.freq_pulse(0, 0, (204.8e6, 0, 819.2e6, 0), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(204.8e6, 0, 819.2e6), shift=0,
                                   ispl=(0x4000000000, 0x1000000000, 0x2000000000))

    inst = Jaqal_v1.freq_pulse(0, 0, (204.8e6, 0, 1638.4e6, 0), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(204.8e6, 0, 1638.4e6), shift=0,
                                   ispl=(0x4000000000, 0x2000000000, 0x4000000000))

    inst = Jaqal_v1.freq_pulse(0, 0, (204.8e6, 0, 409.6e6, 0), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(204.8e6, 0, 409.6e6), shift=1,
                                   ispl=(0x4000000000, 0x1000000000, 0x4000000000))

    inst = Jaqal_v1.freq_pulse(0, 0, (204.8e6, 0, 0, 819.2e6), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(204.8e6, 0, 0, 819.2e6),
                                   shift=0, ispl=(0x4000000000, 0x400000000,
                                                  0x1800000000, 0x1800000000))

    inst = Jaqal_v1.freq_pulse(0, 0, (204.8e6, 0, 0, 409.6e6), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(204.8e6, 0, 0, 409.6e6),
                                   shift=1, ispl=(0x4000000000, 0x400000000,
                                                  0x3000000000, 0x6000000000))

    inst = Jaqal_v1.freq_pulse(0, 0, (204.8e6, 0, 0, 409.6e6), 4096,
                               False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4096, fspl=(204.8e6, 0, 0, 409.6e6),
                                   shift=11,
                                   ispl=(0x4000000000, 0x4000, 0xc000000, 0x6000000000))

    for i in range(2000):
        o0 = random.random() * 100e6
        o1 = random.random() * 200e6 - 100e6
        o2 = random.random() * 200e6 - 100e6
        o3 = random.random() * 200e6 - 100e6
        cycles = random.randint(4, 1000000)
        inst = Jaqal_v1.freq_pulse(0, 0, (o0, o1, o2, o3), cycles,
                                   False, False, False)
        assert inst == MatchParamPulse('freq', cycles=cycles,
                                       fspl=(o0, o1, o2, o3), abs=0.05, rel=1e-10)

def test_amp_spline():
    inst = Jaqal_v1.amp_pulse(0, 0, (1, 0, 0, 0), 1000, False, False, False)
    assert inst == MatchParamPulse('amp', cycles=1000, shift=31,
                                   ispl=(0x7fff800000,), fspl=(1,))

    inst = Jaqal_v1.amp_pulse(0, 0, (-1, 0, 0, 0), 2100, False, False, False)
    assert inst == MatchParamPulse('amp', cycles=2100, shift=31,
                                   ispl=(-0x7fff800000,), fspl=(-1,))

    inst = Jaqal_v1.amp_pulse(0, 0, (-1, 2.0000305180437934, 0, 0), 4,
                              False, False, False)
    assert inst == MatchParamPulse('amp', cycles=4, fspl=(-1, 2.0000305180437934),
                                   shift=0, ispl=(-0x7fff800000, 0x4000000000))

    # Test the exact rounding threshold to make sure we are rounding things correctly.
    inst = Jaqal_v1.amp_pulse(0, 0, (-1, 2.0000000000000004, 0, 0), 4,
                              False, False, False)
    assert inst == MatchParamPulse('amp', cycles=4, fspl=(-1, 2.0000305180437934),
                                   shift=0, ispl=(-0x7fff800000, 0x4000000000))

    inst = Jaqal_v1.amp_pulse(0, 0, (-1, 2, 0, 0), 4, False, False, False)
    assert inst == MatchParamPulse('amp', cycles=4, fspl=(-1, 2), shift=1,
                                   ispl=(-0x7fff800000, 0x7fff800000))

    # Higher orders
    inst = Jaqal_v1.amp_pulse(0, 0, (-1, 0, 2.0000305180437934, 0), 4,
                              False, False, False)
    assert inst == MatchParamPulse('amp', cycles=4, fspl=(-1, 0, 2.0000305180437934),
                                   shift=0, ispl=(-0x7fff800000, 0x1000000000,
                                                  0x2000000000))

    inst = Jaqal_v1.amp_pulse(0, 0, (-1, 0, 2.0000305180437934, 0), 1024,
                              False, False, False)
    assert inst == MatchParamPulse('amp', cycles=1024, fspl=(-1, 0, 2.0000305180437934),
                                   shift=8, ispl=(-0x7fff800000, 0x10000000,
                                                  0x2000000000))

    inst = Jaqal_v1.amp_pulse(0, 0, (1, 0, -2.0000305180437934, 0), 1024,
                              False, False, False)
    assert inst == MatchParamPulse('amp', cycles=1024, fspl=(1, 0, -2.0000305180437934),
                                   shift=8, ispl=(0x7fff800000, -0x10000000,
                                                  -0x2000000000))


    inst = Jaqal_v1.amp_pulse(0, 0, (-1, 0, 0, 2.0000305180437934), 4,
                              False, False, False)
    assert inst == MatchParamPulse('amp', cycles=4, fspl=(-1, 0, 0, 2.0000305180437934),
                                   shift=0, ispl=(-0x7fff800000, 0x400000000,
                                                  0x1800000000, 0x1800000000))

    inst = Jaqal_v1.amp_pulse(0, 0, (-1, 0, 0, 2.0000305180437934), 2048,
                              False, False, False)
    assert inst == MatchParamPulse('amp', cycles=2048, shift=9,
                                   ispl=(-0x7fff800000, 0, 0xc000000, 0x1800000000),
                                   fspl=(-1, 0, 0, 2.0000305180437934), abs=5e-7)

    for i in range(2000):
        o0 = random.random() * 2 - 1
        o1 = random.random() * 2 - 1
        o2 = random.random() * 2 - 1
        o3 = random.random() * 2 - 1
        cycles = random.randint(4, 1000000)
        inst = Jaqal_v1.amp_pulse(0, 0, (o0, o1, o2, o3), cycles,
                                  False, False, False)
        assert inst == MatchParamPulse('amp', cycles=cycles,
                                       fspl=(o0, o1, o2, o3), abs=1e-4, rel=1e-4)

class PhaseTester:
    @staticmethod
    def pulse(spl, cycles):
        return Jaqal_v1.phase_pulse(0, 0, spl, cycles, False, False, False)

    @staticmethod
    def match(**kws):
        return MatchParamPulse('phase', **kws)

class FrameTester:
    @staticmethod
    def pulse(spl, cycles):
        return Jaqal_v1.frame_pulse(0, 0, spl, cycles, False, False, False, 0, 0)

    @staticmethod
    def match(**kws):
        return MatchFramePulse(**kws)

@pytest.mark.parametrize('cls', [PhaseTester, FrameTester])
def test_phase_spline(cls):
    inst = cls.pulse((-0.5, 0, 0, 0), 1000)
    assert inst == cls.match(cycles=1000, shift=31, ispl=(-0x8000000000,), fspl=(-0.5,))

    inst = cls.pulse((0.25, 0, 0, 0), 2100)
    assert inst == cls.match(cycles=2100, shift=31, ispl=(0x4000000000,), fspl=(0.25,))

    inst = cls.pulse((-0.25, 1, 0, 0), 4)
    assert inst == cls.match(cycles=4, shift=0, ispl=(-0x4000000000, 0x4000000000),
                             fspl=(-0.25, 1))

    inst = cls.pulse((-0.25, 2, 0, 0), 4)
    assert inst == cls.match(cycles=4, fspl=(-0.25, -2), shift=0,
                             ispl=(-0x4000000000, -0x8000000000))

    inst = cls.pulse((-0.25, 0.999999999999, 0, 0), 4)
    assert inst == cls.match(cycles=4, shift=0, ispl=(-0x4000000000, 0x4000000000),
                             fspl=(-0.25, 1))

    # Test the exact rounding threshold to make sure we are rounding things correctly.
    inst = cls.pulse((-0.25, 0.9999999999981811, 0, 0), 4)
    assert inst == cls.match(cycles=4, shift=0, ispl=(-0x4000000000, 0x4000000000),
                             fspl=(-0.25, 1))

    inst = cls.pulse((-0.25, 0.999999999998181, 0, 0), 4)
    assert inst == cls.match(cycles=4, shift=1, ispl=(-0x4000000000, 0x7fffffffff),
                             fspl=(-0.25, 0.999999999998181))

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

class ChannelLUTs:
    def __init__(self, chn):
        self.code = b''
        self.chn = chn
        self._pulse_map = {}
        self._seq = []
        self._gate_starts = []
        self._gate_ends = []
        self._gate_map = {}
        self._gates = []
        self.sequence = []

    def write_pulse(self, inst):
        inst_bytes = inst.to_bytes()
        addr = self._pulse_map.get(inst_bytes)
        if addr is None:
            addr = len(self._pulse_map)
            self.code += Jaqal_v1.program_PLUT(inst, addr).to_bytes()
            self._pulse_map[inst_bytes] = addr
        return addr

    def write_gate(self, insts):
        ids = tuple(self.write_pulse(inst) for inst in insts)
        gaddr = self._gate_map.get(ids)
        if gaddr is None:
            start = len(self._seq)
            end = start + len(ids) - 1
            self._seq.extend(ids)
            gaddr = len(self._gate_map)
            self._gate_starts.append(start)
            self._gate_ends.append(end)
            self._gate_map[ids] = gaddr
        return gaddr

    def add_gate(self, insts):
        self._gates.append(self.write_gate(insts))
        self.sequence.extend(insts)

    def finalize(self):
        for i in range(0, len(self._seq), 9):
            s = self._seq[i:i + 9]
            addr = list(range(i, i + len(s)))
            self.code += Jaqal_v1.program_SLUT(self.chn, addr, s).to_bytes()

        for i in range(0, len(self._gate_starts), 6):
            starts = self._gate_starts[i:i + 6]
            ends = self._gate_ends[i:i + 6]
            addr = list(range(i, i + len(starts)))
            self.code += Jaqal_v1.program_GLUT(self.chn, addr, starts, ends).to_bytes()

    def get_gate_seq(self):
        c = b''
        for i in range(0, len(self._gates), 24):
            c += Jaqal_v1.sequence(self.chn, 0, self._gates[i:i + 24]).to_bytes()
        return c

class SequenceConstructor:
    def __init__(self):
        self.channels = [ChannelLUTs(i) for i in range(8)]

    def add_gate(self, insts):
        if not insts:
            return
        chn = insts[0].channel
        for inst in insts:
            assert inst.channel == chn
        self.channels[chn].add_gate(insts)

    def get_inst_bytes(self):
        code = b''
        for c in self.channels:
            c.finalize()
            code += c.code
        for c in self.channels:
            code += c.get_gate_seq()
        return code

    def check_seq(self):
        pulses = Jaqal_v1.extract_pulses(self.get_inst_bytes())
        expected = []
        for c in self.channels:
            expected.extend(c.sequence)
        assert pulses == expected

def rand_pulse(chn):
    tone = random.randint(0, 1)
    cycles = random.randint(10, 11)
    ty = random.randint(0, 3)
    o0 = random.randint(0, 3) / 3
    o1 = random.randint(0, 3) / 3
    o2 = random.randint(0, 3) / 3
    o3 = random.randint(0, 3) / 3
    if ty == 0:
        return Jaqal_v1.freq_pulse(chn, tone, (o0 * 100e6, o1 * 200e6 - 100e6,
                                               o2 * 200e6 - 100e6, o3 * 200e6 - 100e6),
                                   cycles, False, False, False)
    elif ty == 1:
        return Jaqal_v1.amp_pulse(chn, tone, (o0 * 2 - 1, o1 * 2 - 1,
                                              o2 * 2 - 1, o3 * 2 - 1),
                                  cycles, False, False, False)
    elif ty == 2:
        return Jaqal_v1.phase_pulse(chn, tone, (o0 * 2 - 1, o1 * 2 - 1,
                                                o2 * 2 - 1, o3 * 2 - 1),
                                    cycles, False, False, False)
    else:
        return Jaqal_v1.frame_pulse(chn, tone, (o0 * 2 - 1, o1 * 2 - 1,
                                                o2 * 2 - 1, o3 * 2 - 1),
                                    cycles, False, False, False, 0, 0)

def test_extract_pulse():
    pulses = [rand_pulse(random.randint(0, 7)) for _ in range(3000)]
    stream_bytes = b''
    for pulse in pulses:
        stream_bytes += Jaqal_v1.stream(pulse).to_bytes()
    assert Jaqal_v1.extract_pulses(stream_bytes) == pulses

    gseq = Jaqal_v1.sequence(2, 0, [0, 1, 2, 3]).to_bytes()
    wait_anc = Jaqal_v1.sequence(2, 1, [0, 1, 2, 3]).to_bytes()
    cont_anc = Jaqal_v1.sequence(2, 2, [0, 1, 2, 3]).to_bytes()
    glut = Jaqal_v1.program_GLUT(2, [0, 1, 2, 3], [0, 3, 6, 10], [2, 5, 9, 10]).to_bytes()
    glut_inverse = Jaqal_v1.program_GLUT(2, [0, 1, 2, 3],
                                         [2, 5, 9, 10], [0, 3, 6, 10]).to_bytes()
    slut1 = Jaqal_v1.program_SLUT(2, [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                  [0, 1, 2, 3, 0, 1, 2, 3, 0]).to_bytes()
    slut2 = Jaqal_v1.program_SLUT(2, [9, 10, 11], [1, 2, 3]).to_bytes()

    plut_mem = [rand_pulse(2) for _ in range(4)]
    plut = b''.join(Jaqal_v1.program_PLUT(plut_mem[i], i).to_bytes() for i in range(4))

    # Out of bound gate
    assert Jaqal_v1.extract_pulses(gseq) == [invalid_pulse] * 4
    # Out of bound sequence
    assert Jaqal_v1.extract_pulses(glut + gseq) == [invalid_pulse] * 11
    assert Jaqal_v1.extract_pulses(glut_inverse + gseq) == [invalid_pulse] * 4
    # Out of bound pulse
    assert Jaqal_v1.extract_pulses(slut1 + slut2 + glut + gseq) == [invalid_pulse] * 11

    assert Jaqal_v1.extract_pulses(plut + slut1 + slut2 + glut + gseq) == plut_mem * 2 + plut_mem[:3]
    assert Jaqal_v1.extract_pulses(plut + slut1 + slut2 + glut + wait_anc) == [invalid_pulse]
    assert Jaqal_v1.extract_pulses(plut + slut1 + slut2 + glut + cont_anc) == [invalid_pulse]

def test_sequence():
    for _ in range(100):
        sc = SequenceConstructor()
        for _ in range(300):
            chn = random.randint(0, 7)
            sc.add_gate([rand_pulse(chn) for _ in range(random.randint(3, 4))])
        sc.check_seq()


class ChannelGenTester:
    def __init__(self):
        self.gen = JaqalChannelGen_v1()
        self.pulses = []

    def add_pulse(self, pulse, cycle):
        self.gen.add_pulse(pulse, cycle)
        self.pulses.append((cycle, pulse))

    def clear(self):
        self.gen.clear()
        self.pulses.clear()

    def end(self):
        self.gen.end()
        sorted(self.pulses, key=lambda x: x[0])
        plut = self.get_plut()
        slut = self.get_slut()
        glut = self.get_glut()
        gseq = self.get_gseq()
        plut_used = [False] * len(plut)
        slut_used = [False] * len(slut)
        glut_used = [False] * len(glut)
        seq_pulses = []
        def add_plut(pulse_idx, cycle):
            plut_used[pulse_idx] = True
            seq_pulses.append((plut[pulse_idx], cycle))
        def add_slut(idx1, idx2, cycle):
            for idx in range(idx1, idx2 + 1):
                slut_used[idx] = True
                add_plut(slut[idx], cycle if idx == idx1 else None)
        def add_gate(idx, cycle):
            glut_used[idx] = True
            sidx1, sidx2 = glut[idx]
            add_slut(sidx1, sidx2, cycle)
        for (cycle, gidx) in gseq:
            add_gate(gidx, cycle)
        assert len(seq_pulses) == len(self.pulses)
        for ((sp, sc), (cycle, pulse)) in zip(seq_pulses, self.pulses):
            assert sp == pulse
            if sc is not None:
                assert sc == cycle
        assert all(plut_used)
        assert all(slut_used)
        assert all(glut_used)

    def get_plut(self):
        return self.gen.get_plut()

    def get_slut(self):
        return self.gen.get_slut()

    def get_glut(self):
        return self.gen.get_glut()

    def get_gseq(self):
        return self.gen.get_gseq()


def test_channel_gen():
    freq_params = [(0, 0, 0, 0), (100e6, 0, 0, 0),
                   (150e6, 0, 0, 0), (100e6, 0, 100e6, 0),
                   (150e6, -100e6, 0, 0), (150e6, 10e6, 0, -20e6),
                   (3e6, 1e6, 0, -2e6), (3e6, 1e6, 2e6, 0)]
    amp_params = [(0, 0, 0, 0), (1, 0, 0, 0), (0.15, 0, 0, 0), (0.1, 0, 0.1, 0),
                  (0.15, -0.1, 0, 0), (0.15, 0.01, 0, -0.02),
                  (0.3, 0.1, 0, -0.2), (0.3, 0.1, 0.2, 0)]
    phase_params = [(0, 0, 0, 0), (0.9, 0, 0, 0), (0.15, 0, 0, 0), (0.1, 0, 0.1, 0),
                    (0.15, -0.1, 0, 0), (0.15, 0.01, 0, -0.02),
                    (0.3, 0.1, 0, -0.2), (0.3, 0.1, 0.2, 0)]
    def freq_pulse(tone, cycles, param_idx, trig=False, sync=False, ff=False):
        return Jaqal_v1.freq_pulse(0, tone, freq_params[param_idx],
                                   cycles, trig, sync, ff)
    def amp_pulse(tone, cycles, param_idx, trig=False, sync=False, ff=False):
        return Jaqal_v1.amp_pulse(0, tone, amp_params[param_idx],
                                  cycles, trig, sync, ff)
    def phase_pulse(tone, cycles, param_idx, trig=False, sync=False, ff=False):
        return Jaqal_v1.phase_pulse(0, tone, phase_params[param_idx],
                                    cycles, trig, sync, ff)
    def frame_pulse(tone, cycles, param_idx, trig=False, eof=False, clr=False):
        return Jaqal_v1.frame_pulse(0, tone, phase_params[param_idx],
                                    cycles, trig, eof, clr, 0, 0)
    gen = ChannelGenTester()
    gen.add_pulse(freq_pulse(0, 1000, 0), 0)
    gen.add_pulse(amp_pulse(0, 1000, 0), 0)
    gen.add_pulse(phase_pulse(0, 1000, 0), 0)
    gen.add_pulse(frame_pulse(0, 1000, 0), 0)
    gen.end()

    assert gen.get_plut() == [freq_pulse(0, 1000, 0), amp_pulse(0, 1000, 0),
                              phase_pulse(0, 1000, 0), frame_pulse(0, 1000, 0)]
    assert gen.get_slut() == [0, 1, 2, 3]
    assert gen.get_glut() == [(0, 3)]
    assert gen.get_gseq() == [(0, 0)]

    gen.clear()
    gen.end()

    assert gen.get_plut() == []
    assert gen.get_slut() == []
    assert gen.get_glut() == []
    assert gen.get_gseq() == []

    gen.clear()
    gen.add_pulse(freq_pulse(0, 1000, 0), 0)
    gen.add_pulse(amp_pulse(0, 1000, 0), 0)
    gen.add_pulse(phase_pulse(0, 1000, 0), 0)
    gen.add_pulse(frame_pulse(0, 1000, 0), 0)
    gen.add_pulse(freq_pulse(0, 1000, 0), 1000)
    gen.add_pulse(amp_pulse(0, 1000, 0), 1000)
    gen.add_pulse(phase_pulse(0, 1000, 0), 1000)
    gen.add_pulse(frame_pulse(0, 1000, 0), 1000)
    gen.end()

    assert gen.get_plut() == [freq_pulse(0, 1000, 0), amp_pulse(0, 1000, 0),
                              phase_pulse(0, 1000, 0), frame_pulse(0, 1000, 0)]
    assert gen.get_slut() == [0, 1, 2, 3]
    assert gen.get_glut() == [(0, 3)]
    assert gen.get_gseq() == [(0, 0), (1000, 0)]

    gen.clear()
    gen.add_pulse(freq_pulse(0, 1000, 0), 0)
    gen.add_pulse(amp_pulse(0, 1000, 0), 0)
    gen.add_pulse(phase_pulse(0, 1000, 0), 0)
    gen.add_pulse(frame_pulse(0, 1000, 0), 0)
    gen.add_pulse(freq_pulse(0, 1000, 1), 1000)
    gen.add_pulse(amp_pulse(0, 1000, 1), 1000)
    gen.add_pulse(phase_pulse(0, 1000, 1), 1000)
    gen.add_pulse(frame_pulse(0, 1000, 1), 1000)
    gen.add_pulse(freq_pulse(0, 1000, 0), 2000)
    gen.add_pulse(amp_pulse(0, 1000, 0), 2000)
    gen.add_pulse(phase_pulse(0, 1000, 0), 2000)
    gen.add_pulse(frame_pulse(0, 1000, 0), 2000)
    gen.add_pulse(freq_pulse(0, 1000, 1), 3000)
    gen.add_pulse(amp_pulse(0, 1000, 1), 3000)
    gen.add_pulse(phase_pulse(0, 1000, 1), 3000)
    gen.add_pulse(frame_pulse(0, 1000, 1), 3000)
    gen.end()

    assert gen.get_plut() == [freq_pulse(0, 1000, 0), amp_pulse(0, 1000, 0),
                              phase_pulse(0, 1000, 0), frame_pulse(0, 1000, 0),
                              freq_pulse(0, 1000, 1), amp_pulse(0, 1000, 1),
                              phase_pulse(0, 1000, 1), frame_pulse(0, 1000, 1)]
    assert gen.get_slut() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert gen.get_glut() == [(0, 7)]
    assert gen.get_gseq() == [(0, 0), (2000, 0)]

    gen.clear()
    gen.add_pulse(freq_pulse(0, 1000, 0), 0)
    gen.add_pulse(amp_pulse(0, 1000, 0), 0)
    gen.add_pulse(phase_pulse(0, 1000, 0), 0)
    gen.add_pulse(frame_pulse(0, 1000, 0), 0)
    gen.add_pulse(freq_pulse(0, 1000, 1), 1000)
    gen.add_pulse(amp_pulse(0, 1000, 1), 1000)
    gen.add_pulse(phase_pulse(0, 1000, 1), 1000)
    gen.add_pulse(frame_pulse(0, 1000, 1), 1000)
    gen.add_pulse(freq_pulse(0, 1000, 0), 2000)
    gen.add_pulse(amp_pulse(0, 1000, 0), 2000)
    gen.add_pulse(phase_pulse(0, 1000, 0), 2000)
    gen.add_pulse(frame_pulse(0, 1000, 0), 2000)
    gen.add_pulse(freq_pulse(0, 1000, 1), 3000)
    gen.add_pulse(amp_pulse(0, 1000, 1), 3000)
    gen.add_pulse(phase_pulse(0, 1000, 1), 3000)
    gen.add_pulse(frame_pulse(0, 1000, 1), 3000)
    gen.add_pulse(freq_pulse(0, 1000, 0), 4000)
    gen.add_pulse(amp_pulse(0, 1000, 0), 4000)
    gen.add_pulse(phase_pulse(0, 1000, 0), 4000)
    gen.add_pulse(frame_pulse(0, 1000, 0), 4000)
    gen.end()

    assert gen.get_plut() == [freq_pulse(0, 1000, 0), amp_pulse(0, 1000, 0),
                              phase_pulse(0, 1000, 0), frame_pulse(0, 1000, 0),
                              freq_pulse(0, 1000, 1), amp_pulse(0, 1000, 1),
                              phase_pulse(0, 1000, 1), frame_pulse(0, 1000, 1)]
    assert gen.get_slut() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert gen.get_glut() == [(0, 3), (4, 7)]
    assert gen.get_gseq() == [(0, 0), (1000, 1), (2000, 0), (3000, 1), (4000, 0)]

    gen.clear()
    gen.add_pulse(freq_pulse(0, 1000, 0), 0)
    gen.add_pulse(amp_pulse(0, 1000, 0), 0)
    gen.add_pulse(phase_pulse(0, 1000, 1), 0)
    gen.add_pulse(frame_pulse(0, 1000, 0), 0)
    gen.add_pulse(freq_pulse(0, 1000, 1), 1000)
    gen.add_pulse(amp_pulse(0, 1000, 1), 1000)
    gen.add_pulse(phase_pulse(0, 1000, 1), 1000)
    gen.add_pulse(frame_pulse(0, 1000, 1), 1000)
    gen.add_pulse(freq_pulse(0, 1000, 0), 2000)
    gen.add_pulse(amp_pulse(0, 1000, 0), 2000)
    gen.add_pulse(phase_pulse(0, 1000, 2), 2000)
    gen.add_pulse(frame_pulse(0, 1000, 0), 2000)
    gen.add_pulse(freq_pulse(0, 1000, 1), 3000)
    gen.add_pulse(amp_pulse(0, 1000, 1), 3000)
    gen.add_pulse(phase_pulse(0, 1000, 1), 3000)
    gen.add_pulse(frame_pulse(0, 1000, 1), 3000)
    gen.add_pulse(freq_pulse(0, 1000, 0), 4000)
    gen.add_pulse(amp_pulse(0, 1000, 0), 4000)
    gen.add_pulse(phase_pulse(0, 1000, 3), 4000)
    gen.add_pulse(frame_pulse(0, 1000, 0), 4000)
    gen.add_pulse(freq_pulse(0, 1000, 1), 5000)
    gen.add_pulse(amp_pulse(0, 1000, 1), 5000)
    gen.add_pulse(phase_pulse(0, 1000, 1), 5000)
    gen.add_pulse(frame_pulse(0, 1000, 1), 5000)
    gen.add_pulse(freq_pulse(0, 1000, 0), 6000)
    gen.add_pulse(amp_pulse(0, 1000, 0), 6000)
    gen.add_pulse(phase_pulse(0, 1000, 4), 6000)
    gen.add_pulse(frame_pulse(0, 1000, 0), 6000)
    gen.end()

    assert gen.get_plut() == [freq_pulse(0, 1000, 0), amp_pulse(0, 1000, 0),
                              phase_pulse(0, 1000, 1), frame_pulse(0, 1000, 0),
                              freq_pulse(0, 1000, 1), amp_pulse(0, 1000, 1),
                              frame_pulse(0, 1000, 1), phase_pulse(0, 1000, 2),
                              phase_pulse(0, 1000, 3), phase_pulse(0, 1000, 4)]
    assert gen.get_slut() == [3, 4, 5, 2, 6, 0, 1, 0, 1, 2, 9, 3, 7, 8]
    assert gen.get_glut() == [(0, 6), (7, 9), (10, 11), (12, 12), (13, 13)]
    assert gen.get_gseq() == [(0, 1), (0, 0), (2000, 3), (2000, 0),
                              (4000, 4), (4000, 0), (6000, 2)]

    gen.clear()
    gen.add_pulse(freq_pulse(0, 1000, 0), 0)
    gen.add_pulse(amp_pulse(0, 1000, 0), 0)
    gen.add_pulse(phase_pulse(0, 1000, 0), 0)
    gen.add_pulse(frame_pulse(0, 1000, 0), 0)
    gen.add_pulse(freq_pulse(0, 1000, 1), 1000)
    gen.add_pulse(amp_pulse(0, 1000, 1), 1000)
    gen.add_pulse(phase_pulse(0, 1000, 1), 1000)
    gen.add_pulse(frame_pulse(0, 1000, 1), 1000)
    gen.add_pulse(freq_pulse(0, 1000, 2), 2000)
    gen.add_pulse(amp_pulse(0, 1000, 2), 2000)
    gen.add_pulse(phase_pulse(0, 1000, 2), 2000)
    gen.add_pulse(frame_pulse(0, 1000, 2), 2000)
    gen.add_pulse(freq_pulse(0, 1000, 3), 3000)
    gen.add_pulse(amp_pulse(0, 1000, 3), 3000)
    gen.add_pulse(phase_pulse(0, 1000, 3), 3000)
    gen.add_pulse(frame_pulse(0, 1000, 0), 3000)
    gen.add_pulse(freq_pulse(0, 1000, 0), 4000)
    gen.add_pulse(amp_pulse(0, 1000, 0), 4000)
    gen.add_pulse(phase_pulse(0, 1000, 0), 4000)
    gen.add_pulse(frame_pulse(0, 1000, 0), 4000)
    gen.add_pulse(freq_pulse(0, 1000, 1), 5000)
    gen.add_pulse(amp_pulse(0, 1000, 1), 5000)
    gen.add_pulse(phase_pulse(0, 1000, 1), 5000)
    gen.add_pulse(frame_pulse(0, 1000, 1), 5000)
    gen.add_pulse(freq_pulse(0, 1000, 2), 6000)
    gen.add_pulse(amp_pulse(0, 1000, 2), 6000)
    gen.add_pulse(phase_pulse(0, 1000, 2), 6000)
    gen.add_pulse(frame_pulse(0, 1000, 2), 6000)
    gen.add_pulse(freq_pulse(0, 1000, 3), 7000)
    gen.add_pulse(amp_pulse(0, 1000, 3), 7000)
    gen.add_pulse(phase_pulse(0, 1000, 3), 7000)
    gen.add_pulse(frame_pulse(0, 1000, 1), 7000)
    gen.add_pulse(freq_pulse(0, 1000, 0), 8000)
    gen.add_pulse(amp_pulse(0, 1000, 0), 8000)
    gen.add_pulse(phase_pulse(0, 1000, 0), 8000)
    gen.add_pulse(frame_pulse(0, 1000, 0), 8000)
    gen.add_pulse(freq_pulse(0, 1000, 1), 9000)
    gen.add_pulse(amp_pulse(0, 1000, 1), 9000)
    gen.add_pulse(phase_pulse(0, 1000, 1), 9000)
    gen.add_pulse(frame_pulse(0, 1000, 1), 9000)
    gen.add_pulse(freq_pulse(0, 1000, 2), 10000)
    gen.add_pulse(amp_pulse(0, 1000, 2), 10000)
    gen.add_pulse(phase_pulse(0, 1000, 2), 10000)
    gen.add_pulse(frame_pulse(0, 1000, 2), 10000)
    gen.add_pulse(freq_pulse(0, 1000, 3), 11000)
    gen.add_pulse(amp_pulse(0, 1000, 3), 11000)
    gen.add_pulse(phase_pulse(0, 1000, 3), 11000)
    gen.add_pulse(frame_pulse(0, 1000, 2), 11000)
    gen.add_pulse(freq_pulse(0, 1000, 0), 12000)
    gen.add_pulse(amp_pulse(0, 1000, 0), 12000)
    gen.add_pulse(phase_pulse(0, 1000, 0), 12000)
    gen.add_pulse(frame_pulse(0, 1000, 0), 12000)
    gen.add_pulse(freq_pulse(0, 1000, 1), 13000)
    gen.add_pulse(amp_pulse(0, 1000, 1), 13000)
    gen.add_pulse(phase_pulse(0, 1000, 1), 13000)
    gen.add_pulse(frame_pulse(0, 1000, 1), 13000)
    gen.add_pulse(freq_pulse(0, 1000, 2), 14000)
    gen.add_pulse(amp_pulse(0, 1000, 2), 14000)
    gen.add_pulse(phase_pulse(0, 1000, 2), 14000)
    gen.add_pulse(frame_pulse(0, 1000, 2), 14000)
    gen.add_pulse(freq_pulse(0, 1000, 3), 15000)
    gen.add_pulse(amp_pulse(0, 1000, 3), 15000)
    gen.add_pulse(phase_pulse(0, 1000, 3), 15000)
    gen.add_pulse(frame_pulse(0, 1000, 2), 15000)
    gen.end()

    assert gen.get_plut() == [freq_pulse(0, 1000, 0), amp_pulse(0, 1000, 0),
                              phase_pulse(0, 1000, 0), frame_pulse(0, 1000, 0),
                              freq_pulse(0, 1000, 1), amp_pulse(0, 1000, 1),
                              phase_pulse(0, 1000, 1), frame_pulse(0, 1000, 1),
                              freq_pulse(0, 1000, 2), amp_pulse(0, 1000, 2),
                              phase_pulse(0, 1000, 2), frame_pulse(0, 1000, 2),
                              freq_pulse(0, 1000, 3), amp_pulse(0, 1000, 3),
                              phase_pulse(0, 1000, 3)]
    assert gen.get_slut() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    assert gen.get_glut() == [(0, 14), (3, 3), (7, 7), (11, 11)]
    assert gen.get_gseq() == [(0, 0), (3000, 1), (4000, 0), (7000, 2),
                              (8000, 0), (11000, 3), (12000, 0), (15000, 3)]

    gen.clear()
    t = 0
    for pulse_type in range(8):
        for tone in range(2):
            for cycles in range(1000, 8001, 1000):
                for flags in itertools.product((False, True), (False, True),
                                               (False, True)):
                    gen.add_pulse(freq_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(amp_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(phase_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(frame_pulse(tone, cycles, pulse_type, *flags), t)
                    t += cycles
    gen.end()

    assert len(gen.get_plut()) == 8 * 2 * 8 * 4 * 8
    assert gen.get_slut() == list(range(8 * 2 * 8 * 4 * 8))
    assert gen.get_glut() == [(0, 8 * 2 * 8 * 4 * 8 - 1)]
    assert gen.get_gseq() == [(0, 0)]

    gen.clear()
    t = 0
    for pulse_type in range(8):
        for tone in range(2):
            for cycles in range(1000, 8001, 1000):
                for flags in itertools.product((False, True), (False, True),
                                               (False, True)):
                    gen.add_pulse(freq_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(amp_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(phase_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(frame_pulse(tone, cycles, pulse_type, *flags), t)
                    t += cycles
    with pytest.raises(RuntimeError, match="Too many pulses in sequence."):
        gen.add_pulse(freq_pulse(0, 123, 0), t)

    gen.clear()
    t = 0
    for pulse_type in range(8):
        for tone in range(2):
            for cycles in range(1000, 8001, 1000):
                for flags in itertools.product((False, True), (False, True),
                                               (False, True)):
                    gen.add_pulse(freq_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(amp_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(phase_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(frame_pulse(tone, cycles, pulse_type, *flags), t)
                    t += cycles
    gen.add_pulse(freq_pulse(1, 2000, 2), t)
    gen.add_pulse(amp_pulse(1, 2000, 2), t)
    gen.add_pulse(phase_pulse(1, 2000, 2), t)
    gen.add_pulse(frame_pulse(1, 2000, 2), t)
    t += cycles
    gen.end()

    assert len(gen.get_plut()) == 8 * 2 * 8 * 4 * 8
    assert gen.get_slut() == [1312, 1313, 1314, 1315] + list(range(1312)) + list(range(1316, 8 * 2 * 8 * 4 * 8))
    assert gen.get_glut() == [(0, 3), (4, 1315), (1316, 8 * 2 * 8 * 4 * 8 - 1)]
    assert gen.get_gseq() == [(0, 1), (1448000, 0), (1450000, 2), (4608000, 0)]

    gen.clear()
    t = 0
    for pulse_type in range(8):
        for tone in range(2):
            for cycles in range(1000, 8001, 1000):
                for flags in itertools.product((False, True), (False, True),
                                               (False, True)):
                    gen.add_pulse(freq_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(amp_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(phase_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(frame_pulse(tone, cycles, pulse_type, *flags), t)
                    t += cycles
    gen.add_pulse(freq_pulse(1, 2000, 2), t)
    gen.add_pulse(amp_pulse(0, 1000, 3), t)
    gen.add_pulse(phase_pulse(1, 5000, 7), t)
    gen.add_pulse(frame_pulse(0, 7000, 5), t)
    t += cycles
    with pytest.raises(RuntimeError, match="Too many SLUT entries."):
        gen.end()

    gen.clear()
    t = 0
    for pulse_type in range(8):
        for tone in range(2):
            for cycles in range(1000, 8001, 1000):
                for flags in itertools.product((False, True), (False, True),
                                               (False, True)):
                    gen.add_pulse(freq_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(amp_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(phase_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(frame_pulse(tone, cycles, pulse_type, *flags), t)
                    t += cycles
    for pulse_type in range(8):
        for tone in range(2):
            for cycles in range(1000, 8001, 1000):
                for flags in itertools.product((True, False), (True, False),
                                               (True, False)):
                    gen.add_pulse(freq_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(amp_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(phase_pulse(tone, cycles, pulse_type, *flags), t)
                    gen.add_pulse(frame_pulse(tone, cycles, pulse_type, *flags), t)
                    t += cycles
    with pytest.raises(RuntimeError, match="Too many GLUT entries."):
        gen.end()
