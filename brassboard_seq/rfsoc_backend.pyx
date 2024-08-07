# cython: language_level=3

# Copyright (c) 2024 - 2024 Yichao Yu <yyc1992@gmail.com>

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3.0 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not,
# see <http://www.gnu.org/licenses/>.

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.action cimport Action, RampFunction, SeqCubicSpline, \
  new_ramp_buffer, ramp_get_spline_segments, rampbuffer_alloc_input, rampbuffer_eval
from brassboard_seq.event_time cimport EventTime, round_time_int
from brassboard_seq.rtval cimport is_rtval, rt_eval, RuntimeValue
from brassboard_seq.utils cimport pyfloat_from_double, set_global_tracker, \
  PyErr_Format, PyExc_ValueError

from libcpp.map cimport map as cppmap

cimport cython
from cython.operator cimport dereference as deref
from cpython cimport PyDict_GetItemWithError

cdef re # hide import
import re

cdef extern from "src/rfsoc_backend.cpp" namespace "brassboard_seq::rfsoc_backend":
    struct CompileVTable:
        bint (*is_rtval)(object) noexcept
        bint (*is_ramp)(object) noexcept
    void collect_actions(RFSOCBackend ab,
                         CompileVTable vtable, Action, EventTime) except+

    struct RuntimeVTable:
        object (*rt_eval)(object, unsigned)
        int (*rampbuffer_eval_segments)(object, object, object, object,
                                        double **input, double **output) except -1
        double *(*rampbuffer_alloc_input)(object, int) except NULL
        double *(*rampbuffer_eval)(object, object, object, object) except NULL
        bint (*ramp_get_cubic_spline)(object, cubic_spline_t *sp) noexcept

    void generate_tonedata(RFSOCBackend ab, unsigned age, RuntimeVTable vtable) except +

cdef inline bint is_ramp(obj) noexcept:
    return isinstance(obj, RampFunction)

cdef inline bint ramp_get_cubic_spline(_ramp, cubic_spline_t *sp) noexcept:
    if type(_ramp) is not SeqCubicSpline:
        return False
    sp[0] = cubic_spline_t((<SeqCubicSpline>_ramp).order0,
                           (<SeqCubicSpline>_ramp).order1,
                           (<SeqCubicSpline>_ramp).order2,
                           (<SeqCubicSpline>_ramp).order3)
    return True

cdef inline int rampbuffer_eval_segments(_buff, _func, length, oldval,
                                         double **input, double **output) except -1:
    cdef RampBuffer buff = <RampBuffer>_buff
    cdef RampFunction func = <RampFunction>_func
    pts = ramp_get_spline_segments(func, length, oldval)
    if pts is None:
        return 0
    cdef int npts = len(pts)
    pin = rampbuffer_alloc_input(buff, npts * 3 + 4)
    input[0] = pin
    cdef double prev_v = 0
    cdef double v
    pin[0] = 0
    cdef int i
    for i in range(npts):
        v = pts[i]
        pin[3 * i + 1] = v * (1 / 3) + prev_v * (2 / 3)
        pin[3 * i + 2] = v * (2 / 3) + prev_v * (1 / 3)
        pin[3 * i + 3] = v
        prev_v = v
    v = <double?>length
    pin[3 * npts + 1] = v * (1 / 3) + prev_v * (2 / 3)
    pin[3 * npts + 2] = v * (2 / 3) + prev_v * (1 / 3)
    pin[3 * npts + 3] = v
    output[0] = rampbuffer_eval(buff, func, length, oldval)
    return npts * 3 + 4

cdef inline CompileVTable get_compile_vtable() noexcept nogil:
    cdef CompileVTable vt
    vt.is_rtval = is_rtval
    vt.is_ramp = is_ramp
    return vt

cdef inline RuntimeVTable get_runtime_vtable() noexcept nogil:
    cdef RuntimeVTable vt
    vt.rt_eval = <object (*)(object, unsigned)>rt_eval
    vt.rampbuffer_eval_segments = rampbuffer_eval_segments
    vt.rampbuffer_alloc_input = rampbuffer_alloc_input
    vt.rampbuffer_eval = rampbuffer_eval
    vt.ramp_get_cubic_spline = ramp_get_cubic_spline
    return vt

@cython.auto_pickle(False)
cdef class RFSOCOutputGenerator:
    cdef int start(self) except -1:
        pass

    cdef int add_tone_data(self, int channel, int tone, int64_t duration_cycles,
                           cubic_spline_t frequency_hz, cubic_spline_t amplitude,
                           cubic_spline_t phase_rad, output_flags_t flags) except -1:
        pass

    cdef int finish(self) except -1:
        pass

cdef dummy_post_init
def dummy_post_init(self, /):
    pass

@cython.internal
@cython.auto_pickle(False)
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

@cython.auto_pickle(False)
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
    return PyErr_Format(PyExc_ValueError, 'Invalid channel name %U', <PyObject*>name)

cdef match_rfsoc_dds = re.compile('^dds(\\d+)$').match

cdef inline set_dds_delay(RFSOCBackend self, int dds, delay):
    if delay < 0:
        PyErr_Format(PyExc_ValueError, "DDS time offset %S cannot be negative.",
                     <PyObject*>delay)
    if delay > 0.1:
        PyErr_Format(PyExc_ValueError, "DDS time offset %S cannot be more than 100ms.",
                     <PyObject*>delay)
    self.channels.set_dds_delay(dds, round_time_int(delay))

@cython.auto_pickle(False)
@cython.final
cdef class RFSOCBackend:
    def __init__(self, RFSOCOutputGenerator generator, /):
        self.eval_status = False
        self.generator = generator
        self.ramp_buffer = new_ramp_buffer()
        self.rt_dds_delay = {}

    @property
    def has_output(self):
        return self.channels.channels.size() != 0

    def set_dds_delay(self, int dds, delay, /):
        if is_rtval(delay):
            self.rt_dds_delay[dds] = delay
            return
        set_dds_delay(self, dds, delay)

    cdef int finalize(self) except -1:
        bt_guard = set_global_tracker(&self.seq.seqinfo.bt_tracker)
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
        collect_actions(self, get_compile_vtable(), None, None)

    cdef int runtime_finalize(self, unsigned age) except -1:
        bt_guard = set_global_tracker(&self.seq.seqinfo.bt_tracker)
        for dds, delay in self.rt_dds_delay.items():
            set_dds_delay(self, dds, rt_eval(<RuntimeValue>delay, age))
        self.generator.start()
        try:
            generate_tonedata(self, age, get_runtime_vtable())
        finally:
            self.generator.finish()
