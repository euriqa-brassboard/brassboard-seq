# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.action cimport RampFunction, new_ramp_buffer
from brassboard_seq.utils cimport pyfloat_from_double

from libcpp.map cimport map as cppmap

cimport cython
from cython.operator cimport dereference as deref
from cpython cimport PyDict_GetItemWithError, PyErr_Format, PyObject

cdef re # hide import
import re

cdef extern from "src/rfsoc_backend.cpp" namespace "rfsoc_backend":
    pass

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

cdef PyObject *raise_invalid_channel(tuple path) except NULL:
    name = '/'.join(path)
    return PyErr_Format(ValueError, 'Invalid channel name %U', <PyObject*>name)

cdef match_rfsoc_dds = re.compile('^dds(\\d+)$').match

@cython.final
cdef class RFSOCBackend:
    def __init__(self, RFSOCOutputGenerator generator):
        self.generator = generator
        self.ramp_buffer = new_ramp_buffer()

    cdef int finalize(self) except -1:
        # Channel name format: rfsoc/dds<chn>/<tone>/<param>
        cdef cppmap[int, int] chn_idx_map
        cdef int idx = -1
        cdef int ddsnum
        cdef int tonenum
        for _path in self.seq.seqinfo.channel_paths:
            idx += 1
            path = <tuple>_path
            if <str>path[0] != self.prefix:
                continue
            if len(path) != 4:
                raise_invalid_channel(path)
            m = match_rfsoc_dds(path[1])
            if m is None:
                raise_invalid_channel(path)
            ddsnum = <int>int(m[1])
            if ddsnum > 31:
                raise_invalid_channel(path)
            tonenum = <int>int(path[2])
            if tonenum > 1:
                raise_invalid_channel(path)
            chn = (ddsnum << 1) | tonenum
            it = chn_idx_map.find(chn)
            if it == chn_idx_map.end():
                chn_idx = self.channels.add_tone_channel(chn)
                chn_idx_map[chn] = chn_idx
            else:
                chn_idx = deref(it).second
            param = <str>path[3]
            if param == 'freq':
                param_enum = ToneFreq
            elif param == 'phase':
                param_enum = TonePhase
            elif param == 'amp':
                param_enum = ToneAmp
            elif param == 'ff':
                param_enum = ToneFF
            else:
                # Make the C compiler happy since it doesn't know
                # that `raise_invalid_channel` doesn't return
                param_enum = ToneFF
                raise_invalid_channel(path)
            self.channels.add_seq_channel(idx, chn_idx, param_enum)
