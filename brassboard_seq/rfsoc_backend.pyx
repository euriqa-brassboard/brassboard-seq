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
from brassboard_seq.action cimport Action, RampFunction, SeqCubicSpline
from brassboard_seq.event_time cimport EventTime, round_time_f64
from brassboard_seq.rtval cimport is_rtval, rtval_cache, rt_eval_tagval, RuntimeValue
from brassboard_seq.utils cimport pyfloat_from_double, set_global_tracker, \
  PyErr_Format, PyExc_ValueError, assume_not_none, _assume_not_none, py_object

cimport cython
from cython.operator cimport dereference as deref
from cpython cimport PyDict_GetItemWithError, PyTypeObject

cdef re # hide import
import re

cdef extern from "src/rfsoc_backend.cpp" namespace "brassboard_seq::rfsoc_backend":
    PyTypeObject *rtval_type
    PyTypeObject *rampfunction_type
    PyTypeObject *seqcubicspline_type
    void collect_actions(RFSOCBackend ab, Action, EventTime) except+

    void generate_tonedata(RFSOCBackend ab, RuntimeValue, RampFunction,
                           SeqCubicSpline) except +

    object new_tone_data(PulseCompilerInfo info, int channel, int tone,
                         int64_t duration_cycles, cubic_spline_t freq,
                         cubic_spline_t amp, cubic_spline_t phase,
                         output_flags_t flags) except +

rtval_type = <PyTypeObject*>RuntimeValue
rampfunction_type = <PyTypeObject*>RampFunction
seqcubicspline_type = <PyTypeObject*>SeqCubicSpline

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
    cdef list tonedata_field_names
    cdef list tonedata_field_values
    cdef object py_tone0_num
    cdef object py_tone1_num
    cdef list py_channel_nums
    cdef str channel_str
    cdef str tone_str
    cdef str duration_cycles_str
    cdef str frequency_hz_str
    cdef str amplitude_str
    cdef str phase_rad_str
    cdef str frame_rotation_rad_str
    cdef str wait_trigger_str
    cdef str sync_str
    cdef str output_enable_str
    cdef str feedback_enable_str
    cdef str bypass_lookup_tables_str

cdef PulseCompilerInfo pulse_compiler_info
cdef int init_pulse_compiler_info() except -1:
    global pulse_compiler_info
    if pulse_compiler_info is not None:
        return 0
    self = <PulseCompilerInfo>PulseCompilerInfo.__new__(PulseCompilerInfo)
    from pulsecompiler.rfsoc.tones.tonedata import ToneData
    from pulsecompiler.rfsoc.structures.splines import CubicSpline
    from qiskit.pulse import ControlChannel, DriveChannel
    assert issubclass(CubicSpline, tuple)
    self.orig_post_init = ToneData.__post_init__
    self.CubicSpline = CubicSpline
    self.ToneData = ToneData
    ToneData.__post_init__ = dummy_post_init
    dummy_tonedata = ToneData(0, 0, 4, 0, 0, 0)
    ToneData.__post_init__ = self.orig_post_init
    self.tonedata_field_names = []
    self.tonedata_field_values = []
    for k, v in dummy_tonedata.__dict__.items():
        if k in ('channel', 'tone', 'duration_cycles', 'frequency_hz', 'amplitude',
                 'phase_rad', 'frame_rotation_rad', 'wait_trigger', 'sync',
                 'output_enable', 'feedback_enable', 'bypass_lookup_tables'):
            continue
        self.tonedata_field_names.append(k)
        self.tonedata_field_values.append(v)
    channel_list = [ControlChannel(0), ControlChannel(1)]
    for i in range(62):
        channel_list.append(DriveChannel(i))
    self.channel_list = channel_list
    self.cubic_0 = CubicSpline(pyfloat_from_double(0), pyfloat_from_double(0),
                               pyfloat_from_double(0), pyfloat_from_double(0))
    self.py_tone0_num = 0
    self.py_tone1_num = 1
    self.py_channel_nums = list(range(31))
    self.channel_str = 'channel'
    self.tone_str = 'tone'
    self.duration_cycles_str = 'duration_cycles'
    self.frequency_hz_str = 'frequency_hz'
    self.amplitude_str = 'amplitude'
    self.phase_rad_str = 'phase_rad'
    self.frame_rotation_rad_str = 'frame_rotation_rad'
    self.wait_trigger_str = 'wait_trigger'
    self.sync_str = 'sync'
    self.output_enable_str = 'output_enable'
    self.feedback_enable_str = 'feedback_enable'
    self.bypass_lookup_tables_str = 'bypass_lookup_tables'

    pulse_compiler_info = self


@cython.auto_pickle(False)
@cython.final
cdef class PulseCompilerGenerator:
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
        tonedata = new_tone_data(pulse_compiler_info, channel, tone,
                                 duration_cycles, frequency_hz, amplitude,
                                 phase_rad, flags)
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

cdef inline int set_dds_delay(RFSOCBackend self, int dds, double delay) except -1:
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
    def __init__(self, PulseCompilerGenerator generator, /):
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
        collect_actions(self, None, None)

    cdef int runtime_finalize(self, unsigned age, py_object &pyage) except -1:
        bt_guard = set_global_tracker(&self.seq.seqinfo.bt_tracker)
        for dds, delay in self.rt_dds_delay.items():
            rt_eval_tagval(<RuntimeValue>delay, age, pyage)
            set_dds_delay(self, dds, rtval_cache(<RuntimeValue>delay).get[double]())
        self.generator.start()
        try:
            generate_tonedata(self, None, None, None)
        finally:
            self.generator.finish()
