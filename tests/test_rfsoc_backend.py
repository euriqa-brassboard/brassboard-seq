#

import dummy_pulse_compiler
dummy_pulse_compiler.inject()

from brassboard_seq import rfsoc_backend

def test_generator():
    rfsoc_backend.PulseCompilerGenerator()
    rfsoc_backend.PulseCompilerGenerator()
