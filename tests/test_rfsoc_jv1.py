#

from brassboard_seq.rfsoc_backend import Jaqal_v1, JaqalInst_v1

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
        s = str(other)
        m = re.match('^glut\\.([0-7])\\[([0-6])\\]((?: \\[[0-9]+\\]=\\[[0-9]+,[0-9]+\\]){0,6})$', s)
        assert m is not None
        if self.chn is not None:
            assert int(m[1]) == self.chn
        if self.cnt is not None:
            assert int(m[2]) == self.cnt
        gaddrs = []
        starts = []
        ends = []
        for ss in m[3].split():
            sm = re.match('^\\[([0-9]+)\\]=\\[([0-9]+),([0-9]+)\\]$', ss)
            assert sm is not None
            gaddrs.append(int(sm[1]))
            starts.append(int(sm[2]))
            ends.append(int(sm[3]))
        if self.gaddrs is not None:
            assert gaddrs == self.gaddrs
        if self.starts is not None:
            assert starts == self.starts
        if self.ends is not None:
            assert ends == self.ends
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
        s = str(other)
        m = re.match('^slut\\.([0-7])\\[([0-9])\\]((?: \\[[0-9]+\\]=[0-9]+){0,9})$', s)
        assert m is not None
        if self.chn is not None:
            assert int(m[1]) == self.chn
        if self.cnt is not None:
            assert int(m[2]) == self.cnt
        saddrs = []
        paddrs = []
        for ss in m[3].split():
            sm = re.match('^\\[([0-9]+)\\]=([0-9]+)$', ss)
            assert sm is not None
            saddrs.append(int(sm[1]))
            paddrs.append(int(sm[2]))
        if self.saddrs is not None:
            assert saddrs == self.saddrs
        if self.paddrs is not None:
            assert paddrs == self.paddrs
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
        s = str(other)
        m = re.match('^(gseq|wait_anc|cont_anc)\\.([0-7])\\[([0-9]+)\\]((?: [0-9]+)*)$', s)
        assert m is not None
        if self.mode is not None:
            assert m[1] == self.mode
        if self.chn is not None:
            assert int(m[2]) == self.chn
        if self.cnt is not None:
            assert int(m[3]) == self.cnt
        gaddrs = []
        for ss in m[4].split():
            gaddrs.append(int(ss))
        if self.gaddrs is not None:
            assert gaddrs == self.gaddrs
        return True

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
        assert ispl is None or fspl is None
        if fspl is None:
            assert _rel is None
            assert _abs is None
        if ispl is None:
            assert shift is None
        self.ispl = ispl
        self.shift = shift
        self.fspl = fspl
        self.rel = _rel
        self.abs = _abs
        self.approx = approx or _rel is not None or _abs is not None

    def check_prefix(self, other):
        assert isinstance(other, JaqalInst_v1)
        use_str = self.fspl is not None
        s = str(other) if use_str else repr(other)
        m = re.match('^(pulse_data|plut|stream)\\.([0-7]) ((?:\\[[0-9]+\\]=)?)([_a-z]+)([01]) <([0-9]+)> {([^{}]+)}(.*)$', s)
        assert m is not None
        if self.mode is not None:
            assert m[1] == self.mode
        if self.chn is not None:
            assert int(m[2]) == self.chn
        if self.addr is not None:
            assert self.mode == 'plut'
            assert m[3] == f'[{self.addr}]='
        elif m[1] == 'plut':
            assert m[3]
        else:
            assert not m[3]
        assert m[4] == self.param
        if self.tone is not None:
            assert int(m[5]) == self.tone
        if self.cycles is not None:
            assert int(m[6]) == self.cycles

        spl_orders = m[7].split(', ')
        norders = len(spl_orders)
        if self.fspl is not None:
            got = [float(o) for o in spl_orders]
            expected = list(self.fspl)
            if self.approx:
                got = got + [0] * (4 - len(got))
                expected = expected + [0] * (4 - len(expected))
                expected = pytest.approx(expected, abs=self.abs, rel=self.rel)
            else:
                assert len(expected) >= norders
                if len(expected) > norders:
                    assert all(o == 0 for o in expected[norders:])
                    expected = expected[:norders]
            assert got == expected
        elif self.ispl is not None:
            expected = list(self.ispl)
            assert len(expected) >= norders
            if len(expected) > norders:
                assert all(o == 0 for o in expected[norders:])
                expected = expected[:norders]
            got = []
            shift = 0
            for i, so in enumerate(spl_orders):
                so = so.split('>>')
                if so[0] == '0':
                    got.append(0)
                else:
                    assert so[0].startswith('0x')
                    got.append(int(so[0], 0))
                if len(so) == 1:
                    assert shift == 0
                    continue
                if i == 1:
                    shift = int(so[1])
                    continue
                assert i > 0
                assert int(so[1]) == i * shift
            assert got == expected
            if self.shift is not None:
                assert shift == self.shift

        if self.param == 'frame_rot':
            flags = dict(trig=False, eof=False, clr=False)
        else:
            flags = dict(trig=False, sync=False, enable=False, ff=False)

        for flag in m[8].split():
            if flag in flags:
                assert not flags[flag]
                flags[flag] = True
                continue
            if self.param == 'frame_rot':
                if flag.startswith('fwd:'):
                    assert 'fwd' not in flags
                    flags['fwd'] = int(flag[4:])
                    continue
                if flag.startswith('inv:'):
                    assert 'inv' not in flags
                    flags['inv'] = int(flag[4:])
                    continue
            raise ValueError(f'Unknown pulse flag {flag}')

        if self.param == 'frame_rot':
            assert 'fwd' in flags
            assert 'inv' in flags
        return flags

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

def check_invaild(v, msg):
    assert str(Jaqal_v1.dump_insts(v.to_bytes(32, 'little'))) == f'invalid({msg}): {v:0>64x}'

def test_dump_insts():
    assert Jaqal_v1.dump_insts(b'') == ''
    with pytest.raises(ValueError, match="Instruction stream length not a multiple of instruction size"):
        Jaqal_v1.dump_insts(b'x')

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
    for n in range(6):
        for i in range(2000):
            chn = random.randint(0, 7)
            gaddrs = [random.randint(0, 4095) for _ in range(n)]
            starts = [random.randint(0, 4095) for _ in range(n)]
            ends = [random.randint(0, 4095) for _ in range(n)]
            inst = Jaqal_v1.program_GLUT(chn, gaddrs, starts, ends)
            assert inst == MatchGLUT(chn, gaddrs, starts, ends)
            assert str(inst) == repr(inst)
            assert Jaqal_v1.dump_insts(inst.to_bytes()) == str(inst)
            assert Jaqal_v1.dump_insts(inst.to_bytes() + inst.to_bytes()) == str(inst) + '\n' + str(inst)
            assert inst == JaqalInst_v1(inst.to_bytes())

    glut_base = 0x20000000000000000000000000000000000000000000000000000000000000
    for bit in itertools.chain(range(248, 256), range(239, 245), range(223, 229)):
        check_invaild(glut_base | (1 << bit), 'reserved')
    for n in range(7):
        glut_base_n = glut_base | (n << 236)
        for bit in range(n * 36, 220):
            check_invaild(glut_base_n | (1 << bit), 'reserved')
    for n in range(7, 8):
        check_invaild(glut_base | (n << 236), 'glut_oob')

def test_slut():
    with pytest.raises(ValueError, match="Invalid channel number '-1'"):
        Jaqal_v1.program_SLUT(-1, [], [])
    with pytest.raises(ValueError, match="Invalid channel number '8'"):
        Jaqal_v1.program_SLUT(8, [], [])
    with pytest.raises(ValueError, match="Mismatch address length"):
        Jaqal_v1.program_SLUT(3, [1], [2, 3])
    with pytest.raises(ValueError, match="Too many SLUT addresses to program"):
        Jaqal_v1.program_SLUT(0, list(range(10)), list(range(10)))
    for n in range(9):
        for i in range(3000):
            chn = random.randint(0, 7)
            saddrs = [random.randint(0, 4095) for _ in range(n)]
            paddrs = [random.randint(0, 4095) for _ in range(n)]
            inst = Jaqal_v1.program_SLUT(chn, saddrs, paddrs)
            assert inst == MatchSLUT(chn, saddrs, paddrs)
            assert str(inst) == repr(inst)
            assert Jaqal_v1.dump_insts(inst.to_bytes()) == str(inst)
            assert Jaqal_v1.dump_insts(inst.to_bytes() + inst.to_bytes()) == str(inst) + '\n' + str(inst)
            assert inst == JaqalInst_v1(inst.to_bytes())

    slut_base = 0x40000000000000000000000000000000000000000000000000000000000000
    for bit in itertools.chain(range(248, 256), range(240, 245), range(223, 229)):
        check_invaild(slut_base | (1 << bit), 'reserved')
    for n in range(10):
        slut_base_n = slut_base | (n << 236)
        for bit in range(n * 24, 220):
            check_invaild(slut_base_n | (1 << bit), 'reserved')
    for n in range(10, 16):
        check_invaild(slut_base | (n << 236), 'slut_oob')

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
            assert inst == MatchGSEQ(chn, mode_str, gaddrs)
            assert str(inst) == repr(inst)
            assert Jaqal_v1.dump_insts(inst.to_bytes()) == str(inst)
            assert Jaqal_v1.dump_insts(inst.to_bytes() + inst.to_bytes()) == str(inst) + '\n' + str(inst)
            assert inst == JaqalInst_v1(inst.to_bytes())

    gseq_base = 0x80000000000000000000000000000000000000000000000000000000000000
    for bit in itertools.chain(range(248, 256), range(223, 239)):
        check_invaild(gseq_base | (1 << bit), 'reserved')
    for n in range(25):
        gseq_base_n = gseq_base | (n << 239)
        for bit in range(n * 9, 220):
            check_invaild(gseq_base_n | (1 << bit), 'reserved')
    for n in range(25, 64):
        check_invaild(gseq_base | (n << 239), 'gseq_oob')

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

        assert inst == MatchParamPulse(param, chn, tone, 'pulse_data', None, cycles,
                                       trig, sync, enable, ff,
                                       ispl=(0, 0, 0, 0), shift=0)
        assert inst == MatchParamPulse(param, chn, tone, 'pulse_data', None, cycles,
                                       trig, sync, enable, ff, fspl=(0, 0, 0, 0))
        assert plut_inst == MatchParamPulse(param, chn, tone, 'plut', addr, cycles,
                                            trig, sync, enable, ff,
                                            ispl=(0, 0, 0, 0), shift=0)
        assert plut_inst == MatchParamPulse(param, chn, tone, 'plut', addr, cycles,
                                            trig, sync, enable, ff, fspl=(0, 0, 0, 0))
        assert stm_inst == MatchParamPulse(param, chn, tone, 'stream', None, cycles,
                                           trig, sync, enable, ff,
                                           ispl=(0, 0, 0, 0), shift=0)
        assert stm_inst == MatchParamPulse(param, chn, tone, 'stream', None, cycles,
                                           trig, sync, enable, ff, fspl=(0, 0, 0, 0))
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
            check_invaild(vi | (1 << bit), 'reserved')
            check_invaild(stm_vi | (1 << bit), 'reserved')

        for bit in itertools.chain(range(241, 245), range(228, 229), range(223, 224),
                                   range(200, 220)):
            check_invaild(vi | (1 << bit), 'reserved')
            check_invaild(plut_vi | (1 << bit), 'reserved')
            check_invaild(stm_vi | (1 << bit), 'reserved')

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

        vi = int(inst)

        assert inst == MatchFramePulse(chn, tone, 'pulse_data', None, cycles,
                                       trig, eof, clr, fwd, inv,
                                       ispl=(0, 0, 0, 0), shift=0)
        assert inst == MatchFramePulse(chn, tone, 'pulse_data', None, cycles,
                                       trig, eof, clr, fwd, inv, fspl=(0, 0, 0, 0))
        assert plut_inst == MatchFramePulse(chn, tone, 'plut', addr, cycles,
                                            trig, eof, clr, fwd, inv,
                                            ispl=(0, 0, 0, 0), shift=0)
        assert plut_inst == MatchFramePulse(chn, tone, 'plut', addr, cycles,
                                            trig, eof, clr, fwd, inv, fspl=(0, 0, 0, 0))
        assert stm_inst == MatchFramePulse(chn, tone, 'stream', None, cycles,
                                           trig, eof, clr, fwd, inv,
                                           ispl=(0, 0, 0, 0), shift=0)
        assert stm_inst == MatchFramePulse(chn, tone, 'stream', None, cycles,
                                           trig, eof, clr, fwd, inv, fspl=(0, 0, 0, 0))
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
            check_invaild(vi | (1 << bit), 'reserved')
            check_invaild(stm_vi | (1 << bit), 'reserved')

        for bit in itertools.chain(range(241, 245), range(223, 224),
                                   range(200, 218)):
            check_invaild(vi | (1 << bit), 'reserved')
            check_invaild(plut_vi | (1 << bit), 'reserved')
            check_invaild(stm_vi | (1 << bit), 'reserved')

def test_freq_spline():
    inst = Jaqal_v1.freq_pulse(0, 0, (-409.6e6, 0, 0, 0), 1000, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=1000, shift=0,
                                   ispl=(0xffffff8000000000,))
    assert inst == MatchParamPulse('freq', cycles=1000, fspl=(-409.6e6,))

    inst = Jaqal_v1.freq_pulse(0, 0, (204.8e6, 0, 0, 0), 2100, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=2100, shift=0,
                                   ispl=(0x4000000000,))
    assert inst == MatchParamPulse('freq', cycles=2100, fspl=(204.8e6,))

    inst = Jaqal_v1.freq_pulse(0, 0, (-204.8e6, 819.2e6, 0, 0), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, shift=0,
                                   ispl=(0xffffffc000000000, 0x4000000000))
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(-204.8e6, 819.2e6))

    inst = Jaqal_v1.freq_pulse(0, 0, (-204.8e6, 819.2e6 * 2, 0, 0), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, shift=0,
                                   ispl=(0xffffffc000000000, 0xffffff8000000000))
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(-204.8e6, -1638.4e6))

    inst = Jaqal_v1.freq_pulse(0, 0, (-204.8e6, 8.191999999991808e8, 0, 0), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, shift=0,
                                   ispl=(0xffffffc000000000, 0x4000000000))
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(-204.8e6, 819.2e6))

    # Test the exact rounding threshold to make sure we are rounding things correctly.
    inst = Jaqal_v1.freq_pulse(0, 0, (-204.8e6, 8.191999999985099e8, 0, 0), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, shift=0,
                                   ispl=(0xffffffc000000000, 0x4000000000))
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(-204.8e6, 819.2e6))

    inst = Jaqal_v1.freq_pulse(0, 0, (-204.8e6, 8.191999999985098e8, 0, 0), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, shift=1,
                                   ispl=(0xffffffc000000000, 0x7fffffffff))
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(-204.8e6, 819199999.9985099))

    # Higher orders
    inst = Jaqal_v1.freq_pulse(0, 0, (204.8e6, 0, 819.2e6, 0), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, shift=0,
                                   ispl=(0x4000000000, 0x1000000000, 0x2000000000))
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(204.8e6, 0, 819.2e6))

    inst = Jaqal_v1.freq_pulse(0, 0, (204.8e6, 0, 1638.4e6, 0), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, shift=0,
                                   ispl=(0x4000000000, 0x2000000000, 0x4000000000))
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(204.8e6, 0, 1638.4e6))

    inst = Jaqal_v1.freq_pulse(0, 0, (204.8e6, 0, 409.6e6, 0), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, shift=1,
                                   ispl=(0x4000000000, 0x1000000000, 0x4000000000))
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(204.8e6, 0, 409.6e6))

    inst = Jaqal_v1.freq_pulse(0, 0, (204.8e6, 0, 0, 819.2e6), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, shift=0,
                                   ispl=(0x4000000000, 0x400000000,
                                         0x1800000000, 0x1800000000))
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(204.8e6, 0, 0, 819.2e6))

    inst = Jaqal_v1.freq_pulse(0, 0, (204.8e6, 0, 0, 409.6e6), 4, False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4, shift=1,
                                   ispl=(0x4000000000, 0x400000000,
                                         0x3000000000, 0x6000000000))
    assert inst == MatchParamPulse('freq', cycles=4, fspl=(204.8e6, 0, 0, 409.6e6))

    inst = Jaqal_v1.freq_pulse(0, 0, (204.8e6, 0, 0, 409.6e6), 4096,
                               False, False, False)
    assert inst == MatchParamPulse('freq', cycles=4096, shift=11,
                                   ispl=(0x4000000000, 0x4000, 0xc000000, 0x6000000000))
    assert inst == MatchParamPulse('freq', cycles=4096, fspl=(204.8e6, 0, 0, 409.6e6))

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
    assert inst == MatchParamPulse('amp', cycles=1000, shift=0,
                                   ispl=(0x7fff800000,))
    assert inst == MatchParamPulse('amp', cycles=1000, fspl=(1,))

    inst = Jaqal_v1.amp_pulse(0, 0, (-1, 0, 0, 0), 2100, False, False, False)
    assert inst == MatchParamPulse('amp', cycles=2100, shift=0,
                                   ispl=(0xffffff8000800000,))
    assert inst == MatchParamPulse('amp', cycles=2100, fspl=(-1,))

    inst = Jaqal_v1.amp_pulse(0, 0, (-1, 2.0000305180437934, 0, 0), 4,
                              False, False, False)
    assert inst == MatchParamPulse('amp', cycles=4, shift=0,
                                   ispl=(0xffffff8000800000, 0x4000000000))
    assert inst == MatchParamPulse('amp', cycles=4, fspl=(-1, 2.0000305180437934))

    # Test the exact rounding threshold to make sure we are rounding things correctly.
    inst = Jaqal_v1.amp_pulse(0, 0, (-1, 2.0000000000000004, 0, 0), 4,
                              False, False, False)
    assert inst == MatchParamPulse('amp', cycles=4, shift=0,
                                   ispl=(0xffffff8000800000, 0x4000000000))
    assert inst == MatchParamPulse('amp', cycles=4, fspl=(-1, 2.0000305180437934))

    inst = Jaqal_v1.amp_pulse(0, 0, (-1, 2, 0, 0), 4, False, False, False)
    assert inst == MatchParamPulse('amp', cycles=4, shift=1,
                                   ispl=(0xffffff8000800000, 0x7fff800000))
    assert inst == MatchParamPulse('amp', cycles=4, fspl=(-1, 2))

    # Higher orders
    inst = Jaqal_v1.amp_pulse(0, 0, (-1, 0, 2.0000305180437934, 0), 4,
                              False, False, False)
    assert inst == MatchParamPulse('amp', cycles=4, shift=0,
                                   ispl=(0xffffff8000800000, 0x1000000000,
                                         0x2000000000))
    assert inst == MatchParamPulse('amp', cycles=4, fspl=(-1, 0, 2.0000305180437934))

    inst = Jaqal_v1.amp_pulse(0, 0, (-1, 0, 2.0000305180437934, 0), 1024,
                              False, False, False)
    assert inst == MatchParamPulse('amp', cycles=1024, shift=8,
                                   ispl=(0xffffff8000800000, 0x10000000,
                                         0x2000000000))
    assert inst == MatchParamPulse('amp', cycles=1024, fspl=(-1, 0, 2.0000305180437934))

    inst = Jaqal_v1.amp_pulse(0, 0, (1, 0, -2.0000305180437934, 0), 1024,
                              False, False, False)
    assert inst == MatchParamPulse('amp', cycles=1024, shift=8,
                                   ispl=(0x7fff800000, 0xfffffffff0000000,
                                         0xffffffe000000000))
    assert inst == MatchParamPulse('amp', cycles=1024, fspl=(1, 0, -2.0000305180437934))


    inst = Jaqal_v1.amp_pulse(0, 0, (-1, 0, 0, 2.0000305180437934), 4,
                              False, False, False)
    assert inst == MatchParamPulse('amp', cycles=4, shift=0,
                                   ispl=(0xffffff8000800000, 0x400000000,
                                         0x1800000000, 0x1800000000))
    assert inst == MatchParamPulse('amp', cycles=4,
                                   fspl=(-1, 0, 0, 2.0000305180437934))

    inst = Jaqal_v1.amp_pulse(0, 0, (-1, 0, 0, 2.0000305180437934), 2048,
                              False, False, False)
    assert inst == MatchParamPulse('amp', cycles=2048, shift=9,
                                   ispl=(0xffffff8000800000, 0, 0xc000000,
                                         0x1800000000))
    assert inst == MatchParamPulse('amp', cycles=2048,
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
    assert inst == cls.match(cycles=1000, shift=0, ispl=(0xffffff8000000000,))
    assert inst == cls.match(cycles=1000, fspl=(-0.5,))

    inst = cls.pulse((0.25, 0, 0, 0), 2100)
    assert inst == cls.match(cycles=2100, shift=0, ispl=(0x4000000000,))
    assert inst == cls.match(cycles=2100, fspl=(0.25,))

    inst = cls.pulse((-0.25, 1, 0, 0), 4)
    assert inst == cls.match(cycles=4, shift=0, ispl=(0xffffffc000000000, 0x4000000000))
    assert inst == cls.match(cycles=4, fspl=(-0.25, 1))

    inst = cls.pulse((-0.25, 2, 0, 0), 4)
    assert inst == cls.match(cycles=4, shift=0,
                             ispl=(0xffffffc000000000, 0xffffff8000000000))
    assert inst == cls.match(cycles=4, fspl=(-0.25, -2))

    inst = cls.pulse((-0.25, 0.999999999999, 0, 0), 4)
    assert inst == cls.match(cycles=4, shift=0, ispl=(0xffffffc000000000, 0x4000000000))
    assert inst == cls.match(cycles=4, fspl=(-0.25, 1))

    # Test the exact rounding threshold to make sure we are rounding things correctly.
    inst = cls.pulse((-0.25, 0.9999999999981811, 0, 0), 4)
    assert inst == cls.match(cycles=4, shift=0, ispl=(0xffffffc000000000, 0x4000000000))
    assert inst == cls.match(cycles=4, fspl=(-0.25, 1))

    inst = cls.pulse((-0.25, 0.999999999998181, 0, 0), 4)
    assert inst == cls.match(cycles=4, shift=1, ispl=(0xffffffc000000000, 0x7fffffffff))
    assert inst == cls.match(cycles=4, fspl=(-0.25, 0.999999999998181))

    # Higher orders
    inst = cls.pulse((0.25, 0, 1, 0), 4)
    assert inst == cls.match(cycles=4, shift=0,
                             ispl=(0x4000000000, 0x1000000000, 0x2000000000))
    assert inst == cls.match(cycles=4, fspl=(0.25, 0, 1))

    inst = cls.pulse((0.25, 0, 2, 0), 4)
    assert inst == cls.match(cycles=4, shift=0,
                             ispl=(0x4000000000, 0x2000000000, 0x4000000000))
    assert inst == cls.match(cycles=4, fspl=(0.25, 0, 2))

    inst = cls.pulse((0.25, 0, 0.5, 0), 4)
    assert inst == cls.match(cycles=4, shift=1,
                             ispl=(0x4000000000, 0x1000000000, 0x4000000000))
    assert inst == cls.match(cycles=4, fspl=(0.25, 0, 0.5))

    inst = cls.pulse((0.25, 0, 0, 1), 4)
    assert inst == cls.match(cycles=4, shift=0,
                             ispl=(0x4000000000, 0x400000000,
                                   0x1800000000, 0x1800000000))
    assert inst == cls.match(cycles=4, fspl=(0.25, 0, 0, 1))

    inst = cls.pulse((0.25, 0, 0, 0.5), 4)
    assert inst == cls.match(cycles=4, shift=1,
                             ispl=(0x4000000000, 0x400000000,
                                   0x3000000000, 0x6000000000))
    assert inst == cls.match(cycles=4, fspl=(0.25, 0, 0, 0.5))

    inst = cls.pulse((0.25, 0, 0, 0.5), 4096)
    assert inst == cls.match(cycles=4096, shift=11,
                             ispl=(0x4000000000, 0x4000, 0xc000000, 0x6000000000))
    assert inst == cls.match(cycles=4096, fspl=(0.25, 0, 0, 0.5))

    for i in range(2000):
        o0 = random.random() * 0.99 - 0.5
        o1 = random.random() * 0.99 - 0.5
        o2 = random.random() * 0.99 - 0.5
        o3 = random.random() * 0.99 - 0.5
        cycles = random.randint(4, 1000000)
        inst = cls.pulse((o0, o1, o2, o3), cycles)
        assert inst == cls.match(cycles=cycles, fspl=(o0, o1, o2, o3),
                                 abs=1e-10, rel=1e-10)
