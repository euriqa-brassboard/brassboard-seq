# cython: language_level=3

# Copyright (c) 2024 - 2025 Yichao Yu <yyc1992@gmail.com>

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
from brassboard_seq.backend cimport CompiledSeq
from brassboard_seq.config cimport raise_invalid_channel
from brassboard_seq.event_time cimport round_time_f64
from brassboard_seq.rtval cimport is_rtval, rtval_cache, rt_eval_throw, RuntimeValue
from brassboard_seq.seq cimport seq_get_channel_paths
from brassboard_seq.utils cimport PyErr_Format, PyExc_ValueError

cimport cython
from cython.operator cimport dereference as deref

cdef re # hide import
import re

from brassboard_seq._utils import (JaqalInst_v1, Jaqal_v1, JaqalChannelGen_v1,
                                   JaqalInst_v1_3, Jaqal_v1_3)

cdef extern from "src/rfsoc_backend.cpp" namespace "brassboard_seq::rfsoc_backend":
    void collect_actions(RFSOCBackend ab, CompiledSeq&) except+
    void gen_rfsoc_data(RFSOCBackend ab, CompiledSeq&) except +

    Generator *new_pulse_compiler_generator() except +
    cppclass PulseCompilerGen(Generator):
        object get_output()

    Generator *new_jaqal_pulse_compiler_generator() except +
    cppclass JaqalPulseCompilerGen(Generator):
        object get_prefix(int n) except +
        object get_sequence(int n) except +

    Generator *new_jaqalv1_3_stream_generator() except +
    cppclass Jaqalv1_3StreamGen(Generator):
        object get_prefix(int n) except +
        object get_sequence(int n) except +

cdef class RFSOCGenerator:
    pass

@cython.auto_pickle(False)
@cython.final
cdef class PulseCompilerGenerator(RFSOCGenerator):
    def __cinit__(self):
        self.gen.reset(new_pulse_compiler_generator())

    @property
    def output(self):
        return <dict>(<PulseCompilerGen*>self.gen.get()).get_output()

@cython.auto_pickle(False)
@cython.final
cdef class Jaqalv1Generator(RFSOCGenerator):
    def __cinit__(self):
        self.gen.reset(new_jaqal_pulse_compiler_generator())

    def get_prefix(self, n):
        return (<JaqalPulseCompilerGen*>self.gen.get()).get_prefix(n)

    def get_sequence(self, n):
        return (<JaqalPulseCompilerGen*>self.gen.get()).get_sequence(n)

@cython.auto_pickle(False)
@cython.final
cdef class Jaqalv1_3Generator(RFSOCGenerator):
    def __cinit__(self):
        self.gen.reset(new_jaqalv1_3_stream_generator())

    def get_prefix(self, n):
        return (<Jaqalv1_3StreamGen*>self.gen.get()).get_prefix(n)

    def get_sequence(self, n):
        return (<Jaqalv1_3StreamGen*>self.gen.get()).get_sequence(n)

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

    cdef int finalize(self, CompiledSeq &cseq) except -1:
        # Channel name format: rfsoc/dds<chn>/<tone>/<param>
        cdef cppmap[int, int] chn_idx_map
        cdef int idx = -1
        cdef int ddsnum
        cdef int tonenum
        for _path in seq_get_channel_paths(self.seq):
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
        collect_actions(self, cseq)

    cdef int runtime_finalize(self, CompiledSeq &cseq, unsigned age) except -1:
        for dds, delay in self.rt_dds_delay.items():
            rt_eval_throw(<RuntimeValue>delay, age)
            set_dds_delay(self, dds, rtval_cache(<RuntimeValue>delay).get[double]())
        gen_rfsoc_data(self, cseq)
