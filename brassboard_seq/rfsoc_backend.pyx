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
from brassboard_seq.action cimport RampFunction, SeqCubicSpline
from brassboard_seq.event_time cimport EventTime, round_time_f64
from brassboard_seq.rtval cimport is_rtval, rtval_cache, rt_eval_throw, RuntimeValue
from brassboard_seq.utils cimport set_global_tracker, \
  PyErr_Format, PyExc_ValueError, PyExc_TypeError, \
  assume_not_none, _assume_not_none, py_object, \
  ostream, pybytes_ostream, pylong_from_long, pylong_from_longlong

cimport cython
from cython.operator cimport dereference as deref
from cpython cimport PyDict_GetItemWithError, PyTypeObject, \
  PyBytes_GET_SIZE, PyBytes_AS_STRING, PyList_New, PyList_SET_ITEM, Py_INCREF

from libc.string cimport memcpy

cdef re # hide import
import re

cdef extern from "src/rfsoc_backend.cpp" namespace "brassboard_seq::rfsoc_backend":
    PyTypeObject *rampfunction_type
    PyTypeObject *seqcubicspline_type
    void collect_actions(RFSOCBackend ab, EventTime) except+
    void gen_rfsoc_data(RFSOCBackend ab, RampFunction, SeqCubicSpline) except +

    Generator *new_pulse_compiler_generator() except +
    cppclass PulseCompilerGen(Generator):
        PyObject *output

    Generator *new_jaqal_pulse_compiler_generator() except +
    cppclass JaqalPulseCompilerGen(Generator):
        object get_prefix(int n) except +
        object get_sequence(int n) except +

    struct TimedID:
        int64_t time
        int16_t id

    cppclass JaqalInst:
        bytes to_pybytes() except +
        object to_pylong() except +
        int64_t &operator[](unsigned)
        bint operator==(JaqalInst)

    cppclass PulseAllocator:
        cppmap[JaqalInst,int] pulses

    cppclass _Jaqal_v1 "brassboard_seq::rfsoc_backend::Jaqal_v1":
        enum class SeqMode "brassboard_seq::rfsoc_backend::Jaqal_v1::SeqMode":
            GATE
            WAIT_ANC
            CONT_ANC
            STREAM

        @staticmethod
        JaqalInst freq_pulse(int, int, cubic_spline_t sp, int64_t cycles,
                             bint waittrig, bint sync, bint fb_enable)
        @staticmethod
        JaqalInst amp_pulse(int, int, cubic_spline_t sp, int64_t cycles,
                            bint waittrig, bint sync, bint fb_enable)
        @staticmethod
        JaqalInst phase_pulse(int, int, cubic_spline_t sp, int64_t cycles,
                              bint waittrig, bint sync, bint fb_enable)
        @staticmethod
        JaqalInst frame_pulse(int channel, int tone, cubic_spline_t sp, int64_t cycles,
                              bint waittrig, bint apply_at_end, bint rst_frame,
                              int fwd_frame_mask, int inv_frame_mask)
        @staticmethod
        JaqalInst stream(JaqalInst pulse)
        @staticmethod
        JaqalInst program_PLUT(JaqalInst pulse, uint16_t addr)
        @staticmethod
        JaqalInst program_SLUT(uint8_t chn, const uint16_t *saddrs,
                               const uint16_t *paddrs, int n)
        @staticmethod
        JaqalInst program_GLUT(uint8_t chn, const uint16_t *gaddrs,
                               const uint16_t *starts, const uint16_t *ends, int n)
        @staticmethod
        JaqalInst sequence(uint8_t chn, SeqMode m, uint16_t *gaddrs, int n)

        @staticmethod
        int get_chn(const JaqalInst &inst)

        @staticmethod
        void print_inst(ostream &io, const JaqalInst &inst, bint print_float) except +

        @staticmethod
        void print_insts(ostream &io, const char *p, size_t sz,
                         bint print_float) except +

        @staticmethod
        vector[JaqalInst] extract_pulses(const char *p, size_t sz) except +

        @staticmethod
        dict inst_to_dict(const JaqalInst &inst) except +

        cppclass ChannelGen:
            PulseAllocator pulses
            vector[int16_t] slut
            vector[pair[int16_t,int16_t]] glut
            vector[TimedID] gate_ids
            void add_pulse(const JaqalInst &inst, int64_t cycle) except +
            void clear()
            void end() except +

rampfunction_type = <PyTypeObject*>RampFunction
seqcubicspline_type = <PyTypeObject*>SeqCubicSpline

cdef class RFSOCGenerator:
    pass

@cython.auto_pickle(False)
@cython.final
cdef class PulseCompilerGenerator(RFSOCGenerator):
    def __cinit__(self):
        self.gen.reset(new_pulse_compiler_generator())

    @property
    def output(self):
        return <dict>(<PulseCompilerGen*>self.gen.get()).output

@cython.auto_pickle(False)
@cython.final
cdef class Jaqalv1Generator(RFSOCGenerator):
    def __cinit__(self):
        self.gen.reset(new_jaqal_pulse_compiler_generator())

    def get_prefix(self, n):
        return (<JaqalPulseCompilerGen*>self.gen.get()).get_prefix(n)

    def get_sequence(self, n):
        return (<JaqalPulseCompilerGen*>self.gen.get()).get_sequence(n)

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
    def __init__(self, RFSOCGenerator generator, /):
        self.eval_status = False
        self.generator = generator
        self.rt_dds_delay = {}
        self.use_all_channels = False

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
        collect_actions(self, None)

    cdef int runtime_finalize(self, unsigned age, py_object &pyage) except -1:
        bt_guard = set_global_tracker(&self.seq.seqinfo.bt_tracker)
        for dds, delay in self.rt_dds_delay.items():
            rt_eval_throw(<RuntimeValue>delay, age, pyage)
            set_dds_delay(self, dds, rtval_cache(<RuntimeValue>delay).get[double]())
        self.generator.gen.get().start()
        try:
            gen_rfsoc_data(self, None, None)
        finally:
            self.generator.gen.get().end()

cdef cubic_spline_t _to_spline(spline):
    if isinstance(spline, tuple):
        tu = <tuple>spline
        return cubic_spline_t(tu[0], tu[1], tu[2], tu[3])
    ty = type(spline)
    PyErr_Format(PyExc_TypeError, "Invalid spline type '%S'", <PyObject*>ty)

cdef inline int _check_chn(int chn) except -1:
    if chn < 0 or chn > 7:
        PyErr_Format(PyExc_ValueError, "Invalid channel number '%d'", chn)

cdef inline int _check_chn_tone(int chn, int tone) except -1:
    _check_chn(chn)
    if tone < 0 or tone > 1:
        PyErr_Format(PyExc_ValueError, "Invalid tone number '%d'", tone)

cdef inline int _check_inst_cycles(int64_t cycles) except -1:
    if (cycles >> 40) != 0:
        PyErr_Format(PyExc_ValueError, "Invalid cycle count '%lld'", <long long>cycles)

# Debugging/testing tool
@cython.auto_pickle(False)
@cython.no_gc
@cython.internal
cdef class JaqalInstBase:
    cdef JaqalInst inst
    def __init__(self, data=None):
        if data is None:
            return
        cdef int l
        if isinstance(data, bytes):
            l = PyBytes_GET_SIZE(data)
            memcpy(&self.inst[0], PyBytes_AS_STRING(data), min(l, 32))
        else:
            ty = type(data)
            PyErr_Format(PyExc_TypeError, "Invalid type '%S'", <PyObject*>ty)

    def to_bytes(self):
        return self.inst.to_pybytes()

    def __index__(self):
        return self.inst.to_pylong()

    def __eq__(self, other):
        if not isinstance(other, JaqalInstBase):
            return NotImplemented
        if type(self) is not type(other):
            return False
        return self.inst == (<JaqalInstBase>other).inst

@cython.final
cdef class JaqalInst_v1(JaqalInstBase):
    @property
    def channel(self):
        return _Jaqal_v1.get_chn(self.inst)

    def to_dict(self):
        return _Jaqal_v1.inst_to_dict(self.inst)

    def __str__(self):
        cdef pybytes_ostream io
        _Jaqal_v1.print_inst(io, self.inst, True)
        return io.get_buf().decode()

    def __repr__(self):
        cdef pybytes_ostream io
        _Jaqal_v1.print_inst(io, self.inst, False)
        return io.get_buf().decode()

cdef JaqalInst_v1 new_inst_v1(JaqalInst inst):
    self = <JaqalInst_v1>JaqalInst_v1.__new__(JaqalInst_v1)
    self.inst = inst
    return self

@cython.auto_pickle(False)
@cython.no_gc
@cython.final
cdef class Jaqal_v1:
    @staticmethod
    def freq_pulse(int chn, int tone, spline, int64_t cycles, bint waittrig,
                   bint sync, bint fb_enable):
        _check_chn_tone(chn, tone)
        _check_inst_cycles(cycles)
        return new_inst_v1(_Jaqal_v1.freq_pulse(chn, tone, _to_spline(spline), cycles,
                                                waittrig, sync, fb_enable))

    @staticmethod
    def amp_pulse(int chn, int tone, spline, int64_t cycles, bint waittrig,
                  bint sync, bint fb_enable):
        _check_chn_tone(chn, tone)
        _check_inst_cycles(cycles)
        return new_inst_v1(_Jaqal_v1.amp_pulse(chn, tone, _to_spline(spline), cycles,
                                               waittrig, sync, fb_enable))

    @staticmethod
    def phase_pulse(int chn, int tone, spline, int64_t cycles, bint waittrig,
                    bint sync, bint fb_enable):
        _check_chn_tone(chn, tone)
        _check_inst_cycles(cycles)
        return new_inst_v1(_Jaqal_v1.phase_pulse(chn, tone, _to_spline(spline), cycles,
                                                 waittrig, sync, fb_enable))

    @staticmethod
    def frame_pulse(int chn, int tone, spline, int64_t cycles,
                    bint waittrig, bint apply_at_end, bint rst_frame,
                    int fwd_frame_mask, int inv_frame_mask):
        _check_chn_tone(chn, tone)
        _check_inst_cycles(cycles)
        return new_inst_v1(_Jaqal_v1.frame_pulse(chn, tone, _to_spline(spline),
                                                 cycles, waittrig, apply_at_end,
                                                 rst_frame, fwd_frame_mask,
                                                 inv_frame_mask))

    @staticmethod
    def stream(JaqalInst_v1 pulse, /):
        return new_inst_v1(_Jaqal_v1.stream(pulse.inst))

    @staticmethod
    def program_PLUT(JaqalInst_v1 pulse, int addr, /):
        if addr < 0 or addr >= 4096:
            PyErr_Format(PyExc_ValueError, "Invalid address '%d'", addr)
        return new_inst_v1(_Jaqal_v1.program_PLUT(pulse.inst, addr))

    @staticmethod
    def program_SLUT(int chn, _saddrs, _paddrs, /):
        _check_chn(chn)
        cdef uint16_t saddrs[9]
        cdef uint16_t paddrs[9]
        cdef int n = len(_saddrs)
        if len(_paddrs) != n:
            PyErr_Format(PyExc_ValueError, "Mismatch address length")
        if n > 9:
            PyErr_Format(PyExc_ValueError, "Too many SLUT addresses to program")
        for i in range(n):
            saddrs[i] = _saddrs[i]
            paddrs[i] = _paddrs[i]
        return new_inst_v1(_Jaqal_v1.program_SLUT(chn, saddrs, paddrs, n))

    @staticmethod
    def program_GLUT(int chn, _gaddrs, _starts, _ends, /):
        _check_chn(chn)
        cdef uint16_t gaddrs[6]
        cdef uint16_t starts[6]
        cdef uint16_t ends[6]
        cdef int n = len(_gaddrs)
        if len(_starts) != n or len(_ends) != n:
            PyErr_Format(PyExc_ValueError, "Mismatch address length")
        if n > 6:
            PyErr_Format(PyExc_ValueError, "Too many GLUT addresses to program")
        for i in range(n):
            gaddrs[i] = _gaddrs[i]
            starts[i] = _starts[i]
            ends[i] = _ends[i]
        return new_inst_v1(_Jaqal_v1.program_GLUT(chn, gaddrs, starts, ends, n))

    @staticmethod
    def sequence(int chn, int mode, _gaddrs, /):
        _check_chn(chn)
        cdef uint16_t gaddrs[24]
        cdef int n = len(_gaddrs)
        if n > 24:
            PyErr_Format(PyExc_ValueError, "Too many GLUT addresses to sequence")
        for i in range(n):
            gaddrs[i] = _gaddrs[i]
        if (mode != int(_Jaqal_v1.SeqMode.GATE) and
            mode != int(_Jaqal_v1.SeqMode.WAIT_ANC) and
            mode != int(_Jaqal_v1.SeqMode.CONT_ANC)):
            PyErr_Format(PyExc_ValueError, "Invalid sequencing mode %d.", mode)
        return new_inst_v1(_Jaqal_v1.sequence(chn, <_Jaqal_v1.SeqMode>mode, gaddrs, n))

    @staticmethod
    def dump_insts(bytes b, bint print_float=True):
        cdef pybytes_ostream io
        _Jaqal_v1.print_insts(io, PyBytes_AS_STRING(b),
                              PyBytes_GET_SIZE(b), print_float)
        return io.get_buf().decode()

    @staticmethod
    def extract_pulses(bytes b):
        pulses = _Jaqal_v1.extract_pulses(PyBytes_AS_STRING(b), PyBytes_GET_SIZE(b))
        return [new_inst_v1(pulse) for pulse in pulses]

@cython.auto_pickle(False)
@cython.no_gc
@cython.final
cdef class JaqalChannelGen_v1:
    cdef _Jaqal_v1.ChannelGen chn_gen

    def add_pulse(self, JaqalInst_v1 pulse, int64_t cycle, /):
        self.chn_gen.add_pulse(pulse.inst, cycle)

    def clear(self):
        self.chn_gen.clear()

    def end(self):
        self.chn_gen.end()

    def get_plut(self):
        sz = self.chn_gen.pulses.pulses.size()
        cdef list res = PyList_New(sz)
        for entry in self.chn_gen.pulses.pulses:
            inst = new_inst_v1(entry.first)
            Py_INCREF(inst)
            PyList_SET_ITEM(res, entry.second, inst)
        return res

    def get_slut(self):
        sz = self.chn_gen.slut.size()
        cdef list res = PyList_New(sz)
        for i in range(sz):
            v = pylong_from_long(self.chn_gen.slut[i])
            Py_INCREF(v)
            PyList_SET_ITEM(res, i, v)
        return res

    def get_glut(self):
        sz = self.chn_gen.glut.size()
        cdef list res = PyList_New(sz)
        for i in range(sz):
            gate = self.chn_gen.glut[i]
            v = (pylong_from_long(gate.first), pylong_from_long(gate.second))
            Py_INCREF(v)
            PyList_SET_ITEM(res, i, v)
        return res

    def get_gseq(self):
        sz = self.chn_gen.gate_ids.size()
        cdef list res = PyList_New(sz)
        for i in range(sz):
            gate = self.chn_gen.gate_ids[i]
            v = (pylong_from_longlong(gate.time), pylong_from_long(gate.id))
            Py_INCREF(v)
            PyList_SET_ITEM(res, i, v)
        return res
