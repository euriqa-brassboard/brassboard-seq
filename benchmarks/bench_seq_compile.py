#

import dummy_artiq
dummy_artiq.inject()
import dummy_pulse_compiler
dummy_pulse_compiler.inject()

from brassboard_seq.action import RampFunction, \
    Blackman, BlackmanSquare, LinearRamp, SeqCubicSpline
from brassboard_seq.artiq_backend import ArtiqBackend
from brassboard_seq.backend import SeqCompiler
from brassboard_seq.config import Config
from brassboard_seq.rfsoc_backend import PulseCompilerGenerator, Jaqalv1Generator, \
    RFSOCBackend
from brassboard_seq.rtval import inv, ifelse, RTProp, RuntimeValue
from brassboard_seq.scan import ParamPack, get_param
from brassboard_seq.seq import Seq

import numpy as np

m_2pi = np.pi * 2
m_pi = np.pi

class Blackman2(RampFunction):
    def __init__(self, amp, offset=0):
        super().__init__(amp=amp, offset=amp)

    def eval(self, t, length, oldval, /):
        theta = t * (m_2pi / length) - m_pi
        cost = np.cos(theta)
        val = self.amp * (0.34 + cost * (0.5 + 0.16 * cost))
        val = ifelse(length == 0, t * 0, val)
        return val + self.offset

conf = Config()
conf.add_supported_prefix('artiq')
conf.add_supported_prefix('rfsoc')

def test(n):
    s = Seq(conf)
    comp = SeqCompiler(s)
    # rfsoc_gen = PulseCompilerGenerator()
    rfsoc_gen = Jaqalv1Generator()
    rfsoc = RFSOCBackend(rfsoc_gen)
    comp.add_backend('rfsoc', rfsoc)

    rtios = np.ndarray((0,), np.int32)
    artiq = ArtiqBackend(dummy_artiq.DummyDaxSystem(), rtios)
    comp.add_backend('artiq', artiq)

    ch1 = s.get_channel_id('artiq/ttl1')
    ch2 = s.get_channel_id('artiq/ttl2')

    for i in range(n):
        s.add_step(1).set(ch1, True).set('rfsoc/dds0/1/amp', Blackman(0.8))
        s.add_step(1).pulse(ch2, True).set('rfsoc/dds0/1/freq', 100e6 + 30e6 / n)

    comp.finalize()
    comp.runtime_finalize(1)

def test2(n, m):
    for i in range(m):
        test(n)

test2(1000, 1000)
