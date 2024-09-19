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
  ramp_get_spline_segments, ramp_interp_eval
from brassboard_seq.event_time cimport EventTime, round_time_f64
from brassboard_seq.rtval cimport is_rtval, rt_eval_tagval, RuntimeValue, TagVal
from brassboard_seq.utils cimport pyfloat_from_double, set_global_tracker, \
  PyErr_Format, PyExc_ValueError, assume_not_none, _assume_not_none, py_object, \
  _PyObject_Vectorcall, Py_True, Py_False

from libcpp.map cimport map as cppmap

cimport cython
from cython.operator cimport dereference as deref
from cpython cimport PyDict_GetItemWithError, PyList_GET_ITEM

cdef re # hide import
import re

cdef extern from "src/rfsoc_backend.cpp" namespace "brassboard_seq::rfsoc_backend":
    struct CompileVTable:
        bint (*is_rtval)(object) noexcept
        bint (*is_ramp)(object) noexcept
    void collect_actions(RFSOCBackend ab,
                         CompileVTable vtable, Action, EventTime) except+

    struct RuntimeVTable:
        int (*rt_eval_tagval)(object, unsigned, py_object&) except -1
        bint (*ramp_get_cubic_spline)(object, cubic_spline_t *sp) noexcept
        TagVal (*ramp_interp_eval)(object, double) noexcept

    void generate_tonedata(RFSOCBackend ab, unsigned age, py_object&,
                           RuntimeVTable vtable, RuntimeValue, RampFunction) except +

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

cdef inline CompileVTable get_compile_vtable() noexcept nogil:
    cdef CompileVTable vt
    vt.is_rtval = is_rtval
    vt.is_ramp = is_ramp
    return vt

ctypedef int (*rt_eval_tagval_t)(object, unsigned, py_object&) except -1

cdef inline RuntimeVTable get_runtime_vtable() noexcept nogil:
    cdef RuntimeVTable vt
    vt.rt_eval_tagval = <rt_eval_tagval_t>rt_eval_tagval
    vt.ramp_get_cubic_spline = ramp_get_cubic_spline
    vt.ramp_interp_eval = <TagVal (*)(object, double) noexcept>ramp_interp_eval
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

cdef inline new_cubic_spline(cubic_spline_t sp):
    if sp == cubic_spline_t(0, 0, 0, 0):
        return pulse_compiler_info.cubic_0
    o0 = pyfloat_from_double(sp.order0)
    o1 = pyfloat_from_double(sp.order1)
    o2 = pyfloat_from_double(sp.order2)
    o3 = pyfloat_from_double(sp.order3)
    cdef PyObject *args[4]
    args[0] = <PyObject*>o0
    args[1] = <PyObject*>o1
    args[2] = <PyObject*>o2
    args[3] = <PyObject*>o3
    return _PyObject_Vectorcall(<PyObject*>pulse_compiler_info.CubicSpline,
                                args, 4, NULL)

cdef tonedata_kwarg_names = ('bypass_lookup_tables',)
cdef py_tone0_num = 0
cdef py_tone1_num = 1
cdef py_channel_nums = list(range(31))

cdef inline new_tone_data(int channel, int tone, int64_t duration_cycles,
                          sp_freq, sp_amp, sp_phase, output_flags_t flags):
    cdef PyObject *args[12]
    args[0] = PyList_GET_ITEM(py_channel_nums, channel)
    args[1] = (<PyObject*>py_tone1_num) if tone == 1 else (<PyObject*>py_tone0_num)
    cdef py_duration = duration_cycles
    args[2] = <PyObject*>py_duration
    args[3] = <PyObject*>sp_freq
    args[4] = <PyObject*>sp_amp
    args[5] = <PyObject*>sp_phase
    args[6] = <PyObject*>pulse_compiler_info.cubic_0
    args[7] = Py_True if flags.wait_trigger else Py_False
    args[8] = Py_True if flags.sync else Py_False
    args[9] = Py_False
    args[10] = Py_True if flags.feedback_enable else Py_False
    args[11] = Py_False
    return _PyObject_Vectorcall(<PyObject*>pulse_compiler_info.ToneData,
                                args, 11, <PyObject*>tonedata_kwarg_names)


@cython.auto_pickle(False)
@cython.final
cdef class PulseCompilerGenerator(RFSOCOutputGenerator):
    cdef readonly dict output

    def __init__(self):
        init_pulse_compiler_info()
        self.output = {}

    cdef int start(self) except -1:
        pulse_compiler_info.ToneData.__post_init__ = dummy_post_init
        _assume_not_none(<void*>self.output)
        self.output.clear()

    cdef int add_tone_data(self, int channel, int tone, int64_t duration_cycles,
                           cubic_spline_t frequency_hz, cubic_spline_t amplitude,
                           cubic_spline_t phase_rad, output_flags_t flags) except -1:
        sp_freq = new_cubic_spline(frequency_hz)
        sp_amp = new_cubic_spline(amplitude)
        sp_phase = new_cubic_spline(phase_rad)
        tonedata = new_tone_data(channel, tone, duration_cycles,
                                 sp_freq, sp_amp, sp_phase, flags)
        _assume_not_none(<void*>pulse_compiler_info.channel_list)
        key = pulse_compiler_info.channel_list[(channel << 1) | tone]
        output = <void*>self.output
        ptonedatas = PyDict_GetItemWithError(<dict>output, key)
        cdef list tonedatas
        if ptonedatas != NULL:
            tonedatas = <list>ptonedatas
        else:
            tonedatas = []
            _assume_not_none(output)
            (<dict>output)[key] = tonedatas
        assume_not_none(tonedatas)
        tonedatas.append(tonedata)

    cdef int finish(self) except -1:
        pulse_compiler_info.ToneData.__post_init__ = pulse_compiler_info.orig_post_init

cdef PyObject *raise_invalid_channel(tuple path) except NULL:
    name = '/'.join(path)
    return PyErr_Format(PyExc_ValueError, 'Invalid channel name %U', <PyObject*>name)

cdef match_rfsoc_dds = re.compile('^dds(\\d+)$').match

cdef inline set_dds_delay(RFSOCBackend self, int dds, double delay):
    if delay < 0:
        py_delay = <object>delay
        PyErr_Format(PyExc_ValueError, "DDS time offset %S cannot be negative.",
                     <PyObject*>py_delay)
    if delay > 0.1:
        py_delay = <object>delay
        PyErr_Format(PyExc_ValueError, "DDS time offset %S cannot be more than 100ms.",
                     <PyObject*>py_delay)
    self.channels.set_dds_delay(dds, round_time_f64(delay))

@cython.auto_pickle(False)
@cython.final
cdef class RFSOCBackend:
    def __init__(self, RFSOCOutputGenerator generator, /):
        self.eval_status = False
        self.generator = generator
        self.rt_dds_delay = {}

    @property
    def has_output(self):
        return self.channels.channels.size() != 0

    def set_dds_delay(self, int dds, delay, /):
        if is_rtval(delay):
            self.rt_dds_delay[dds] = delay
            return
        set_dds_delay(self, dds, <double>delay)

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

    cdef int runtime_finalize(self, unsigned age, py_object &pyage) except -1:
        bt_guard = set_global_tracker(&self.seq.seqinfo.bt_tracker)
        for dds, delay in self.rt_dds_delay.items():
            rt_eval_tagval(<RuntimeValue>delay, age, pyage)
            set_dds_delay(self, dds, (<RuntimeValue>delay).cache.get[double]())
        self.generator.start()
        try:
            generate_tonedata(self, age, pyage, get_runtime_vtable(), None, None)
        finally:
            self.generator.finish()
