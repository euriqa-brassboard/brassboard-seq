#

import typing
import types
import sys
import dataclasses as dc

def new_module(parent, name):
    m = types.ModuleType(name)
    if parent is not None:
        setattr(parent, name, m)
    return m

qiskit = new_module(None, 'qiskit')
pulse = new_module(qiskit, 'pulse')

class ControlChannel:
    def __init__(self, index):
        self.index = index

    def __str__(self):
        return f'ControlChannel({self.index})'

    def __repr__(self):
        return str(self)

class DriveChannel:
    def __init__(self, index):
        self.index = index

    def __str__(self):
        return f'DriveChannel({self.index})'

    def __repr__(self):
        return str(self)

pulse.ControlChannel = ControlChannel
pulse.DriveChannel = DriveChannel

pulsecompiler = new_module(None, 'pulsecompiler')
rfsoc = new_module(pulsecompiler, 'rfsoc')

structures = new_module(rfsoc, 'structures')
splines = new_module(structures, 'splines')

tones = new_module(rfsoc, 'tones')
tonedata = new_module(tones, 'tonedata')

IntOrFloat = typing.Union[int, float]

class CubicSpline(typing.NamedTuple):
    order0: IntOrFloat
    order1: IntOrFloat = 0
    order2: IntOrFloat = 0
    order3: IntOrFloat = 0

SplineOrFloat = typing.Union[CubicSpline, float]

@dc.dataclass(frozen=True, order=True)
class ToneData:
    channel: int
    tone: int
    duration_cycles: int
    frequency_hz: SplineOrFloat
    amplitude: SplineOrFloat
    phase_rad: SplineOrFloat
    frame_rotation_rad: SplineOrFloat = 0.0
    wait_trigger: bool = False
    sync: bool = True
    output_enable: bool = True
    feedback_enable: bool = False
    bypass_lookup_tables: bool = True
    frame_rotate_at_end: bool = False
    reset_frame: bool = False

    def __post_init__(self):
        raise RuntimeError("Do not use!!")

splines.CubicSpline = CubicSpline
tonedata.ToneData = ToneData

def inject():
    sys.modules['qiskit'] = qiskit
    sys.modules['qiskit.pulse'] = pulse
    sys.modules['pulsecompiler'] = pulsecompiler
    sys.modules['pulsecompiler.rfsoc'] = rfsoc
    sys.modules['pulsecompiler.rfsoc.structures'] = structures
    sys.modules['pulsecompiler.rfsoc.structures.splines'] = splines
    sys.modules['pulsecompiler.rfsoc.tones'] = tones
    sys.modules['pulsecompiler.rfsoc.tones.tonedata'] = tonedata
