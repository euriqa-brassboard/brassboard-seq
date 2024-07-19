# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.utils cimport pyfloat_from_double

cimport cython
from cpython cimport PyDict_GetItemWithError

cdef class RFSOCOutputGenerator:
    cdef int start(self) except -1:
        pass

    cdef int add_tone_data(self, int channel, int tone, int64_t duration_cycles,
                           cubic_spline_t frequency_hz, cubic_spline_t amplitude,
                           cubic_spline_t phase_rad, output_flags_t flags) except -1:
        pass

    cdef int finish(self) except -1:
        pass

cdef _dummy_post_init(self):
    pass

cdef dummy_post_init = _dummy_post_init

@cython.final
@cython.no_gc
cdef class PulseCompilerInfo:
    cdef list channel_list
    cdef object orig_post_init
    cdef object CubicSpline
    cdef object ToneData
    cdef object cubic_0

cdef PulseCompilerInfo pulse_compiler_info
cdef int init_pulse_compiler_info() except -1:
    global pulse_compiler_info
    if pulse_compiler_info is not None:
        return 0
    self = <PulseCompilerInfo>PulseCompilerInfo.__new__(PulseCompilerInfo)
    from pulsecompiler.rfsoc.tones.tonedata import ToneData
    from pulsecompiler.rfsoc.structures.splines import CubicSpline
    from qiskit.pulse import ControlChannel, DriveChannel
    self.orig_post_init = ToneData.__post_init__
    self.CubicSpline = CubicSpline
    self.ToneData = ToneData
    channel_list = [ControlChannel(0), ControlChannel(1)]
    for i in range(62):
        channel_list.append(DriveChannel(i))
    self.channel_list = channel_list
    self.cubic_0 = CubicSpline(pyfloat_from_double(0), pyfloat_from_double(0),
                               pyfloat_from_double(0), pyfloat_from_double(0))
    pulse_compiler_info = self

@cython.final
cdef class PulseCompilerGenerator(RFSOCOutputGenerator):
    cdef readonly dict output

    def __init__(self):
        init_pulse_compiler_info()
        self.output = {}

    cdef int start(self) except -1:
        pulse_compiler_info.ToneData.__post_init__ = dummy_post_init
        self.output.clear()

    cdef int add_tone_data(self, int channel, int tone, int64_t duration_cycles,
                           cubic_spline_t frequency_hz, cubic_spline_t amplitude,
                           cubic_spline_t phase_rad, output_flags_t flags) except -1:
        if frequency_hz == cubic_spline_t(0, 0, 0, 0):
            sp_freq = pulse_compiler_info.cubic_0
        else:
            sp_freq = pulse_compiler_info.CubicSpline(
                pyfloat_from_double(frequency_hz.order0),
                pyfloat_from_double(frequency_hz.order1),
                pyfloat_from_double(frequency_hz.order2),
                pyfloat_from_double(frequency_hz.order3))
        if amplitude == cubic_spline_t(0, 0, 0, 0):
            sp_amp = pulse_compiler_info.cubic_0
        else:
            sp_amp = pulse_compiler_info.CubicSpline(
                pyfloat_from_double(amplitude.order0),
                pyfloat_from_double(amplitude.order1),
                pyfloat_from_double(amplitude.order2),
                pyfloat_from_double(amplitude.order3))
        if phase_rad == cubic_spline_t(0, 0, 0, 0):
            sp_phase = pulse_compiler_info.cubic_0
        else:
            sp_phase = pulse_compiler_info.CubicSpline(
                pyfloat_from_double(phase_rad.order0),
                pyfloat_from_double(phase_rad.order1),
                pyfloat_from_double(phase_rad.order2),
                pyfloat_from_double(phase_rad.order3))
        tonedata = pulse_compiler_info.ToneData(
            channel, tone, duration_cycles, sp_freq, sp_amp, sp_phase,
            pulse_compiler_info.cubic_0, flags.wait_trigger, flags.sync, False,
            flags.feedback_enable, bypass_lookup_tables=False)
        key = pulse_compiler_info.channel_list[(channel << 1) | tone]
        ptonedatas = PyDict_GetItemWithError(self.output, key)
        cdef list tonedatas
        if ptonedatas != NULL:
            tonedatas = <list>ptonedatas
        else:
            tonedatas = []
            self.output[key] = tonedatas
        tonedatas.append(tonedata)

    cdef int finish(self) except -1:
        pulse_compiler_info.ToneData.__post_init__ = pulse_compiler_info.orig_post_init
